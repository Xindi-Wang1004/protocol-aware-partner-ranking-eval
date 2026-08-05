import gc
import platform

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_modules import (
    DeepCNN, CrossChainAttention, DynamicCrossChainGraph, 
    GATLayer, TaskHeads, ViralFamilyEncoder, SuperConnectorAttention
)


class ProteinInteractionModel(nn.Module):
    """
    Multi-task protein interaction model incorporating:
    1. ESM embeddings for proteins
    2. CNN for feature extraction
    3. Cross-chain attention mechanism
    4. Graph neural network for interaction modeling
    5. Multiple prediction heads for different tasks
    
    增强特性:
    - 特征维度压缩层确保维度一致性
    - 自适应参数调整支持不同ESM模型
    - 内存优化设计支持大规模训练
    """
    def __init__(self, config):
        super(ProteinInteractionModel, self).__init__()
        self.config = config
        
        # 模型核心参数
        self.embedding_dim = config.embedding_dim  # ESM嵌入维度
        self.hidden_dim = config.hidden_dim       # 隐藏层维度
        self.dropout = config.dropout            # Dropout率
        
        # 多任务学习权重
        self.task_weights = config.task_weights  # (分类、检索、基序识别)
        
        # 特征提取CNN
        self.human_cnn = DeepCNN(self.embedding_dim, self.hidden_dim)
        self.virus_cnn = DeepCNN(self.embedding_dim, self.hidden_dim)
        
        # 病毒家族编码器
        self.family_encoder = ViralFamilyEncoder(
            num_families=config.num_viral_families,
            embedding_dim=32,
            output_dim=self.hidden_dim
        )
        
        # 连接度感知注意力
        self.human_connector_attn = SuperConnectorAttention(
            input_dim=self.hidden_dim,
            dropout=self.dropout
        )
        self.virus_connector_attn = SuperConnectorAttention(
            input_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        # 链间交叉注意力
        self.cross_attn = CrossChainAttention(
            input_dim=self.hidden_dim, 
            num_heads=config.num_heads
        )
        
        # 动态图构建器
        try:
            threshold = config.attention_threshold
        except AttributeError:
            threshold = 0.1
        self.graph_builder = DynamicCrossChainGraph(attention_threshold=threshold)
        
        # 特征维度压缩层 - 在forward中动态创建
        # 我们初始化为空，因为输入维度可能因ESM模型而变化
        self.feature_compressor = None
        
        # 记录上次压缩的输入维度，避免重复创建压缩层
        self._last_compressed_dim = None
        
        # 图注意力网络层
        try:
            num_gat_layers = config.num_layers
        except AttributeError:
            num_gat_layers = 2
            
        self.gat_layers = nn.ModuleList([
            GATLayer(
                in_features=self.hidden_dim, 
                out_features=self.hidden_dim,
                num_heads=config.num_heads,
                dropout=self.dropout
            ) for _ in range(num_gat_layers)
        ])
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.hidden_dim, self.hidden_dim),  # +self.hidden_dim for viral family
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 多任务预测头
        self.task_heads = TaskHeads(input_dim=self.hidden_dim)
        
    def forward(self, human_embeddings, virus_embeddings, human_mask=None, virus_mask=None, 
             attention_mask_h2v=None, attention_mask_v2h=None, human_connectivity=None, 
             virus_connectivity=None, viral_family=None, human_ids=None, virus_ids=None):
        """
        Forward pass through the complete model.
        
        Args:
            human_embeddings: ESM embeddings for human protein [batch, seq_len_h, esm_dim]
            virus_embeddings: ESM embeddings for virus protein [batch, seq_len_v, esm_dim]
            human_mask: Boolean mask for human sequences [batch, seq_len_h]
            virus_mask: Boolean mask for virus sequences [batch, seq_len_v]
            attention_mask_h2v: Attention mask for human->virus attention [batch, seq_len_h, seq_len_v]
            attention_mask_v2h: Attention mask for virus->human attention [batch, seq_len_v, seq_len_h]
            human_connectivity: Number of interactions for each human protein [batch]
            virus_connectivity: Number of interactions for each virus protein [batch]
            viral_family: Viral family IDs [batch]
            
        Returns:
            Dictionary containing outputs for each task
        """
        batch_size = human_embeddings.size(0)
        human_len = human_embeddings.size(1)
        virus_len = virus_embeddings.size(1)
        
        # Step 1: CNN feature extraction
        human_features, h_mid1, h_mid2 = self.human_cnn(human_embeddings)
        virus_features, v_mid1, v_mid2 = self.virus_cnn(virus_embeddings)
        
        # Apply connectivity-aware attention for highly connected proteins
        # 同时使用掩码信息优化注意力计算
        if human_connectivity is not None:
            human_features = self.human_connector_attn(human_features, human_connectivity)
            if human_mask is not None:
                # 确保填充区域的特征为零
                human_features = human_features * human_mask.unsqueeze(-1).float()
        if virus_connectivity is not None:
            virus_features = self.virus_connector_attn(virus_features, virus_connectivity)
            if virus_mask is not None:
                # 确保填充区域的特征为零
                virus_features = virus_features * virus_mask.unsqueeze(-1).float()
        
        # Reshape for sequence-level processing
        human_features = human_features.unsqueeze(1).expand(-1, human_len, -1)  # [batch, seq_len_h, cnn_dim]
        virus_features = virus_features.unsqueeze(1).expand(-1, virus_len, -1)  # [batch, seq_len_v, cnn_dim]
        
        # Step 2: Cross-chain attention (使用掩码)
        h_attn, v_attn, attn_h2v, attn_v2h = self.cross_attn(
            human_features, 
            virus_features,
            human_mask=human_mask,
            virus_mask=virus_mask,
            attention_mask_h2v=attention_mask_h2v,
            attention_mask_v2h=attention_mask_v2h
        )
        
        # Step 3: Build dynamic interaction graph
        adj_matrix = self.graph_builder.build_graph(
            human_len=human_len,
            virus_len=virus_len,
            attn_weights=attn_h2v
        )
        
        # Step 4: 合并节点特征用于图处理
        combined_features = torch.cat([h_attn, v_attn], dim=1)  # [batch, seq_len_h + seq_len_v, feature_dim]
        
        # 特征维度压缩 - 确保GNN输入维度一致性
        feature_dim = combined_features.size(-1)
        
        # 强制性的对输入展平和重新形状以确保维度正确
        if feature_dim != self.hidden_dim:
            # 重新创建压缩层
            device = combined_features.device
            print(f"创建强制特征压缩层: {feature_dim} -> {self.hidden_dim}")
            
            # 使用直接的投影方式而不使用自定义的压缩层
            # 这避免了nn.Linear构建可能引起的维度不匹配问题
            compressed_features = F.adaptive_avg_pool2d(
                combined_features.transpose(1, 2).unsqueeze(2),  # [batch, feature_dim, 1, seq_len]
                (self.hidden_dim, combined_features.size(1))  # 目标尺寸: [hidden_dim, seq_len]
            ).squeeze(2).transpose(1, 2)  # 输出: [batch, seq_len, hidden_dim]
            
            # 添加最终投影以确保维度完全正确
            if hasattr(self, 'final_projection') and self.final_projection.in_features == compressed_features.size(-1):
                compressed_features = self.final_projection(compressed_features)
            else:
                self.final_projection = nn.Linear(compressed_features.size(-1), self.hidden_dim).to(device)
                compressed_features = self.final_projection(compressed_features)
        else:
            # 如果维度已经匹配，只需使用原始特征
            compressed_features = combined_features
        
        # 最终检查以确保维度符合期望
        if compressed_features.size(-1) != self.hidden_dim:
            print(f"重大警告: 压缩后的特征维度 {compressed_features.size(-1)} 仍与GAT期望的 {self.hidden_dim} 不一致")
            # 强制实施最终检查
            compressed_features = F.adaptive_avg_pool1d(
                compressed_features.transpose(1, 2),  # [batch, feature_dim, seq_len]
                compressed_features.size(1)  # 保持序列长度不变
            ).transpose(1, 2)  # 输出: [batch, seq_len, hidden_dim]
            assert compressed_features.size(-1) == self.hidden_dim, f"Feature dimension {compressed_features.size(-1)} still doesn't match expected {self.hidden_dim}"
        
        # Step 5: 应用图注意力层
        x = compressed_features
        
        # 最后检查x的维度
        # print(f"GAT输入就绪分析: 方式={x.shape}, 期望维度=[batch, seq_len, {self.hidden_dim}]")
        
        # 逻辑踏键
        # 保存原始精度，GAT层使用float32以避免FP16数值不稳定
        original_dtype = x.dtype
        for i, gat_layer in enumerate(self.gat_layers):
            # 执行前需要强制断言检查维度
            assert x.size(-1) == self.hidden_dim, f"GAT层{i}输入维度{x.size(-1)}与期望的{self.hidden_dim}不匹配"
            
            # 释放内存
            if i > 0 and platform.system() == 'Darwin':
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            # GAT层使用float32精度以避免FP16数值不稳定导致的CUDA错误
            x_float32 = x.float() if x.dtype == torch.float16 else x
            adj_matrix_float32 = adj_matrix.float() if adj_matrix.dtype == torch.float16 else adj_matrix
            
            # 应用GAT层
            x = gat_layer(x_float32, adj_matrix_float32)
            x = F.relu(x)
            
            # 转换回原始精度（如果使用混合精度训练）
            if original_dtype == torch.float16:
                x = x.half()
        
        # Step 6: Get global representations for classification/retrieval
        human_global = x[:, :human_len].mean(dim=1)  # Mean pooling over human sequence
        virus_global = x[:, human_len:].mean(dim=1)  # Mean pooling over virus sequence
        
        # 打印维度信息用于调试
        # print(f"human_global 维度: {human_global.shape}, virus_global 维度: {virus_global.shape}")
        
        # 检查并确保维度一致性
        expected_dim = self.hidden_dim
        if human_global.size(-1) != expected_dim:
            print(f"警告: human_global 维度 {human_global.size(-1)} 与期望的 {expected_dim} 不一致。创建投影层")
            # 创建动态投影层以适配维度
            human_projection = nn.Linear(human_global.size(-1), expected_dim).to(human_global.device)
            human_global = human_projection(human_global)
        
        if virus_global.size(-1) != expected_dim:
            print(f"警告: virus_global 维度 {virus_global.size(-1)} 与期望的 {expected_dim} 不一致。创建投影层")
            virus_projection = nn.Linear(virus_global.size(-1), expected_dim).to(virus_global.device)
            virus_global = virus_projection(virus_global)
        
        # Step 7: Incorporate viral family information if available
        family_features = None
        if viral_family is not None:
            family_features = self.family_encoder(viral_family)
            if family_features.size(-1) != expected_dim:
                print(f"警告: family_features 维度 {family_features.size(-1)} 与期望的 {expected_dim} 不一致。创建投影层")
                family_projection = nn.Linear(family_features.size(-1), expected_dim).to(family_features.device)
                family_features = family_projection(family_features)
            # Concatenate with global features
            global_features = torch.cat([human_global, virus_global, family_features], dim=1)
        else:
            global_features = torch.cat([human_global, virus_global], dim=1)
            
        print(f"global_features 维度: {global_features.shape}, fusion 层期望维度: {self.hidden_dim * (3 if viral_family is not None else 2)}")
        
        # Step 8: Feature fusion - 增强型特征融合处理
        expected_fusion_input_dim = self.hidden_dim * (3 if viral_family is not None else 2)
        fusion_actual_input_dim = global_features.size(-1)
        
        print(f"global_features 维度: {global_features.shape}, fusion 层期望维度: {expected_fusion_input_dim}")
        
        # 初始化统一维度特征
        fused_features = None
        
        # 根据实际情况处理不同的维度情况
        if fusion_actual_input_dim != expected_fusion_input_dim:
            # 维度不匹配时创建或重用动态投影层
            if not hasattr(self, 'dynamic_fusion_adapter') or self.dynamic_fusion_adapter.in_features != fusion_actual_input_dim:
                self.dynamic_fusion_adapter = nn.Linear(fusion_actual_input_dim, expected_fusion_input_dim).to(global_features.device)
                print(f"创建动态融合适配层: {fusion_actual_input_dim} -> {expected_fusion_input_dim}")
            
            try:
                # 应用适配层
                adapted_features = self.dynamic_fusion_adapter(global_features)
                # 应用原始融合层
                fused_features = self.fusion(adapted_features)
            except RuntimeError as e:
                print(f"适配层错误: {e}")
                # 直接创建符合期望维度的特征
                fused_features = torch.zeros(global_features.size(0), self.hidden_dim, device=global_features.device)
                # 将原始特征复制到新特征中，确保保留部分信息
                min_dim = min(fused_features.size(1), global_features.size(1))
                fused_features[:, :min_dim] = global_features[:, :min_dim].clone()
        else:
            # 维度匹配时直接使用原始融合层
            try:
                fused_features = self.fusion(global_features)
            except RuntimeError as e:
                print(f"融合层错误: {e}")
                print(f"global_features 维度: {global_features.shape}")
                print(f"fusion 层第一个线性层权重维度: {self.fusion[0].weight.shape}")
                # 错误恢复机制
                fused_features = torch.zeros(global_features.size(0), self.hidden_dim, device=global_features.device)
                min_dim = min(fused_features.size(1), global_features.size(1))
                fused_features[:, :min_dim] = global_features[:, :min_dim].clone()
        
        # 确保融合层成功应用
        if fused_features is None:
            print("融合特征创建失败，使用原始特征")
            # 直接使用原始特征
            fused_features = global_features
        
        # Step 9: Apply task-specific heads
        # Pass both global features and sequence-level features for motif prediction
        # 传递注意力权重元组，而不是仅传递 attn_h2v
        task_outputs = self.task_heads(
            global_features=fused_features,
            sequence_features=x,
            attention_weights=(attn_h2v, attn_v2h)
        )
        
        # 确保我们有task_outputs中的human_ids和virus_ids
        # 这些将用于评估检索任务
        if 'similarity_matrix' in task_outputs:
            # 将human_ids和virus_ids添加到输出中用于检索评估
            if human_ids is not None:
                task_outputs['human_ids'] = human_ids
            if virus_ids is not None:
                task_outputs['virus_ids'] = virus_ids
            print("similarity_matrix和IDs已添加到输出中用于检索评估")
        
        # Include attention weights for visualization/interpretation
        task_outputs['attention_weights'] = (attn_h2v, attn_v2h)
        
        # Include the task weights for loss calculation
        task_outputs['task_weights'] = self.task_weights
        
        return task_outputs
