import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeepCNN(nn.Module):
    """
    Deep Convolutional Neural Network module for extracting features from protein embeddings.
    Uses multi-scale convolutional layers to capture different length sequence patterns.
    Supports dynamic input and output dimensions.
    """
    def __init__(self, input_dim=1280, output_dim=64):
        super(DeepCNN, self).__init__()
        
        # 计算中间层大小，确保合理的维度逼近output_dim
        # 设计三个逐渐减小的层次，从输入维度逼近输出维度
        decay_factor = (input_dim / output_dim) ** (1/3)  # 对于1536->64，每层减少约3.4倍
        mid_dim1 = int(input_dim / decay_factor)
        mid_dim2 = int(mid_dim1 / decay_factor)
        
        print(f"创建CNN层: input_dim={input_dim}, mid_dim1={mid_dim1}, mid_dim2={mid_dim2}, output_dim={output_dim}")
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_dim, mid_dim1, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_dim1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_dim1, mid_dim2, kernel_size=5, padding=2),
            nn.BatchNorm1d(mid_dim2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_dim2, output_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # 全局最大池化
        )
        
        # 保存重要维度信息便于调试
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x: Tensor of shape [batch_size, seq_length, embedding_dim]
            
        Returns:
            Tuple containing:
                - out: Tensor of shape [batch_size, output_dim]
                - x1: First level features
                - x2: Second level features
        """
        # Reshape for Conv1D (Conv1D expects [batch, channels, length])
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_length]
        
        # Pass through the convolutional blocks
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        x3 = self.conv_block3(x2)  # [batch_size, output_dim, 1]
        
        # Squeeze the last dimension
        out = x3.squeeze(-1)  # [batch_size, output_dim]
        
        # Also store intermediate representations for multi-level features
        return out, x1, x2  


class CrossChainAttention(nn.Module):
    """
    Cross-chain attention mechanism to model interactions between human and virus proteins.
    Implements multi-head self-attention to capture complex interaction patterns.
    """
    def __init__(self, input_dim=64, num_heads=8, dropout=0.1):
        super(CrossChainAttention, self).__init__()
        self.num_heads = num_heads
        # 存储初始特征维度以在需要时重建组件
        self.init_feature_dim = input_dim
        
        # 确保特征维度能被注意力头数整除
        assert input_dim % num_heads == 0, "Feature dimension must be divisible by number of heads"
        self.head_dim = input_dim // num_heads
        
        # 多头注意力组件
        self._build_attention_components(input_dim, num_heads, dropout)
        
    def forward(self, human_features, virus_features, human_mask=None, virus_mask=None, attention_mask_h2v=None, attention_mask_v2h=None):
        """
        Compute cross-attention between human and virus features.
        
        Args:
            human_features: Tensor of shape [batch_size, seq_len_h, feature_dim]
            virus_features: Tensor of shape [batch_size, seq_len_v, feature_dim]
            human_mask: Boolean mask for human sequences [batch_size, seq_len_h]
            virus_mask: Boolean mask for virus sequences [batch_size, seq_len_v]
            attention_mask_h2v: Attention mask for human->virus attention [batch_size, seq_len_h, seq_len_v]
            attention_mask_v2h: Attention mask for virus->human attention [batch_size, seq_len_v, seq_len_h]
            
        Returns:
            Tuple of (human_output, virus_output, attn_weights_h2v, attn_weights_v2h)
        """
        batch_size = human_features.size(0)
        
        # Human -> Virus attention
        h2v_output, attn_weights_h2v = self._attention_block(
            human_features, virus_features, virus_features,
            attention_mask=attention_mask_h2v
        )
        
        # Virus -> Human attention
        v2h_output, attn_weights_v2h = self._attention_block(
            virus_features, human_features, human_features,
            attention_mask=attention_mask_v2h
        )
        
        return h2v_output, v2h_output, attn_weights_h2v, attn_weights_v2h
    
    def _build_attention_components(self, feature_dim, num_heads, dropout):
        """构建注意力模块的所有组件"""
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        # 多头注意力组件
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # 层正规化和前馈网络
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
    
    def _attention_block(self, query, key, value, attention_mask=None):
        """
        Implementation of the attention mechanism.
        
        Args:
            query: Tensor of shape [batch_size, seq_len_q, feature_dim]
            key: Tensor of shape [batch_size, seq_len_k, feature_dim]
            value: Tensor of shape [batch_size, seq_len_v, feature_dim]
            attention_mask: Optional boolean mask of shape [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        actual_feature_dim = query.size(-1)
        
        # 检查并处理特征维度不匹配的情况
        if actual_feature_dim != self.feature_dim:
            if not hasattr(self, '_dim_warning_shown'):
                print(f"警告: CrossChainAttention期望特征维度{self.feature_dim}，但收到{actual_feature_dim}")
                self._dim_warning_shown = True
            
            # 确保新维度可以被注意力头数整除以避免错误
            if actual_feature_dim % self.num_heads != 0:
                # 使用最接近的可被整除的维度
                new_dim = ((actual_feature_dim // self.num_heads) + 1) * self.num_heads
                print(f"调整特征维度从{actual_feature_dim}到{new_dim}，以便被{self.num_heads}整除")
                
                # 创建一个添加维度的投影层
                projection = nn.Linear(actual_feature_dim, new_dim).to(query.device)
                query = projection(query)
                key = projection(key)
                value = projection(value)
                actual_feature_dim = new_dim
            
            # 重建所有注意力组件以适应新维度
            device = query.device
            self._build_attention_components(actual_feature_dim, self.num_heads, self.dropout.p)
            # 移动所有组件到正确的设备
            self.query_proj = self.query_proj.to(device)
            self.key_proj = self.key_proj.to(device)
            self.value_proj = self.value_proj.to(device)
            self.output_proj = self.output_proj.to(device)
            self.norm1 = self.norm1.to(device)
            self.norm2 = self.norm2.to(device)
            self.ffn = self.ffn.to(device)
        
        residual = query
        
        # 线性投影
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 将Boolean掩码转换为浮点掩码，False -> -inf，True -> 0.0
            # 需要广播到所有注意力头
            # [batch_size, seq_len_q, seq_len_k] -> [batch_size, 1, seq_len_q, seq_len_k]
            mask = attention_mask.unsqueeze(1).float()
            # 使用FP16兼容的值（-1e4而不是-1e9，避免FP16溢出）
            mask = mask.masked_fill(~mask.bool(), -1e4)  # 非掩码区域填充为大的负值
            
            # 应用掩码到注意力分数
            scores = scores + mask
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, actual_feature_dim)
        
        # Apply output projection
        out = self.output_proj(out)
        out = self.dropout(out)
        
        # First residual connection and layer norm
        out = self.norm1(residual + out)
        
        # Feed-forward network
        residual2 = out
        out = self.ffn(out)
        
        # Second residual connection and layer norm
        out = self.norm2(residual2 + out)
        
        return out, attn_weights


class DynamicCrossChainGraph(nn.Module):
    """
    Creates a dynamic graph based on attention weights between human and virus proteins.
    """
    def __init__(self, attention_threshold=0.1):
        super(DynamicCrossChainGraph, self).__init__()
        self.threshold = attention_threshold
        
    def build_graph(self, human_len, virus_len, attn_weights, window_size=5):
        """
        Build a graph adjacency matrix based on attention weights and sequence distance.
        
        Args:
            human_len: Length of human protein sequence
            virus_len: Length of virus protein sequence
            attn_weights: Attention weights between human and virus proteins
            window_size: Window size for sequence neighborhood connections
            
        Returns:
            Adjacency matrix for the graph
        """
        batch_size = attn_weights.size(0)
        
        # Create internal connections based on sequence distance
        h_adj = self._create_sequence_adj(human_len, window_size).to(attn_weights.device)
        v_adj = self._create_sequence_adj(virus_len, window_size).to(attn_weights.device)
        
        # Create cross-chain connections based on attention weights
        # Average across attention heads
        if len(attn_weights.shape) == 4:  # [batch, heads, seq_h, seq_v]
            attn_weights = attn_weights.mean(dim=1)  # [batch, seq_h, seq_v]
        
        # Apply threshold to focus on significant interactions
        cross_adj = (attn_weights > self.threshold).float()
        
        # Build complete adjacency matrix
        total_len = human_len + virus_len
        adj = torch.zeros(batch_size, total_len, total_len, device=attn_weights.device)
        
        # Fill adjacency matrix blocks
        adj[:, :human_len, :human_len] = h_adj.unsqueeze(0).expand(batch_size, -1, -1)
        adj[:, human_len:, human_len:] = v_adj.unsqueeze(0).expand(batch_size, -1, -1)
        adj[:, :human_len, human_len:] = cross_adj
        adj[:, human_len:, :human_len] = cross_adj.transpose(1, 2)
        
        return adj
    
    def _create_sequence_adj(self, seq_len, window_size):
        """
        Create adjacency matrix for internal protein connections based on sequence distance.
        
        Args:
            seq_len: Length of protein sequence
            window_size: Window size for neighborhood connections
            
        Returns:
            Adjacency matrix for the sequence
        """
        adj = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            adj[i, start:end] = 1
        return adj


class GATLayer(nn.Module):
    """
    Graph Attention Network layer for processing protein graph representations.
    This is a simplified implementation that doesn't depend on external GNN libraries.
    With dynamic feature dimension adaptation.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, alpha=0.2):
        super(GATLayer, self).__init__()
        # 存储原始特征维度以便后续参考
        self.init_in_features = in_features
        self.init_out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        # 创建参数和激活函数
        self._build_parameters(in_features, out_features)
        
    def _build_parameters(self, in_features, out_features):
        """构建 GAT 层的参数"""
        self.in_features = in_features
        self.out_features = out_features
        
        # 定义可学习参数
        self.W = nn.Parameter(torch.zeros(size=(self.num_heads, in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(self.num_heads, 2 * out_features, 1)))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        
        # Leaky ReLU 激活
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 记录参数是否已初始化
        self._parameters_initialized = True
    
    def forward(self, h, adj):
        """
        Forward pass of the GAT layer with dynamic feature dimension adaptation.
        
        Args:
            h: Node features of shape [batch_size, num_nodes, in_features]
            adj: Adjacency matrix of shape [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features of shape [batch_size, num_nodes, out_features] (取多头注意力的平均值)
        """
        batch_size, N = h.size(0), h.size(1)
        actual_in_features = h.size(2)  # 获取实际输入维度
        
        # 检查输入特征维度是否与初始化维度一致
        if actual_in_features != self.in_features:
            if not hasattr(self, '_dim_warning_shown'):
                print(f"警告: GATLayer期望输入维度{self.in_features}，但收到{actual_in_features}")
                self._dim_warning_shown = True
                
            # 重新构建参数以适应新维度
            device = h.device
            self._build_parameters(actual_in_features, self.out_features)
            self.W = self.W.to(device)
            self.a = self.a.to(device)
            
        # 应用线性变换到每个注意力头
        # [batch, heads, nodes, out_features]
        Wh = torch.stack([torch.matmul(h, self.W[i]) for i in range(self.num_heads)], dim=1)
        
        # 内存优化的注意力计算：避免创建巨大的中间张量
        # 使用分块计算或直接计算注意力分数
        # 对于长序列，使用更内存高效的方式
        
        # 检查序列长度，如果太长则使用分块计算
        if N > 500:  # 对于长序列使用内存优化版本
            # 分块计算注意力分数，避免创建 N*N 的大张量
            # 根据序列长度和batch_size动态调整chunk_size（非常保守，避免OOM）
            # batch_size=2时需要更小的chunk_size以避免内存溢出
            if batch_size >= 2:
                # batch_size>=2时使用更小的chunk_size
                if N > 2000:
                    chunk_size = 2  # 超长序列使用极小的chunk
                elif N > 1500:
                    chunk_size = 3
                elif N > 1200:
                    chunk_size = 4  # 超长序列使用非常小的chunk
                elif N > 1000:
                    chunk_size = 5  # 降低chunk_size以应对batch_size=2的内存压力
                elif N > 700:
                    chunk_size = 6
                else:
                    chunk_size = 8  # 降低默认chunk_size
            else:
                # batch_size=1时可以使用稍大的chunk_size
                if N > 2000:
                    chunk_size = 4  # 超长序列使用极小的chunk
                elif N > 1500:
                    chunk_size = 6
                elif N > 1200:
                    chunk_size = 8  # 超长序列使用非常小的chunk
                elif N > 1000:
                    chunk_size = 8  # 降低chunk_size以应对内存压力
                elif N > 700:
                    chunk_size = 10
                else:
                    chunk_size = 15  # 降低默认chunk_size
            
            e_list = []
            
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                chunk_Wh_i = Wh[:, :, i:end_i, :]  # [batch, heads, chunk_i, out_features]
                
                e_chunk_list = []
                for j in range(0, N, chunk_size):
                    end_j = min(j + chunk_size, N)
                    chunk_Wh_j = Wh[:, :, j:end_j, :]  # [batch, heads, chunk_j, out_features]
                    
                    # 计算这个chunk的注意力分数（使用原始GAT公式）
                    # 优化：避免使用expand，直接使用repeat以节省内存
                    chunk_Wh_i_expanded = chunk_Wh_i.unsqueeze(3)  # [batch, heads, chunk_i, 1, out_features]
                    chunk_Wh_j_expanded = chunk_Wh_j.unsqueeze(2)  # [batch, heads, 1, chunk_j, out_features]
                    
                    # 创建组合特征（使用repeat代替expand以节省内存）
                    chunk_i_len = chunk_Wh_i.size(2)
                    chunk_j_len = chunk_Wh_j.size(2)
                    # 使用更安全的方式创建组合特征，避免内存访问错误
                    try:
                        chunk_combinations = torch.cat([
                            chunk_Wh_i_expanded.repeat(1, 1, 1, chunk_j_len, 1),
                            chunk_Wh_j_expanded.repeat(1, 1, chunk_i_len, 1, 1)
                        ], dim=-1)
                    except RuntimeError as e:
                        # 如果内存不足，先同步并清理缓存
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        # 使用更小的chunk重试
                        raise e
                    
                    # 立即释放中间变量并同步
                    del chunk_Wh_i_expanded, chunk_Wh_j_expanded
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # 计算注意力分数
                    chunk_e = torch.matmul(
                        chunk_combinations.view(batch_size, self.num_heads, -1, 2 * self.out_features), 
                        self.a
                    ).squeeze(-1)
                    chunk_e = chunk_e.view(batch_size, self.num_heads, end_i - i, end_j - j)
                    e_chunk_list.append(chunk_e)
                    
                    # 立即释放内存
                    del chunk_combinations, chunk_e, chunk_Wh_j
                    # batch_size>=2时需要更频繁地清理内存
                    if batch_size >= 2 or N > 400:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            # batch_size>=2时每个chunk后都同步，确保内存释放
                            if batch_size >= 2:
                                torch.cuda.synchronize()
                
                e_list.append(torch.cat(e_chunk_list, dim=-1))
                del e_chunk_list, chunk_Wh_i
                # 每行chunk计算完后清理内存（batch_size=1或超长序列时很重要）
                if batch_size == 1 or N > 400:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            e = torch.cat(e_list, dim=2)  # [batch, heads, N, N]
            del e_list
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # 注意：必须保留Wh，因为后面计算h_prime时还需要使用
        else:
            # 对于短序列，使用原始方法
            Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=2)  # [batch, heads, nodes*nodes, out_features]
            Wh_repeated_alternating = Wh.repeat(1, 1, N, 1)  # [batch, heads, nodes*nodes, out_features]
            
            # Concatenate features for attention computation
            all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
            all_combinations = all_combinations.view(batch_size, self.num_heads, N, N, 2 * self.out_features)
            
            # 确保注意力权重向量维度匹配
            if all_combinations.size(-1) != self.a.size(1):
                device = all_combinations.device
                actual_dim = all_combinations.size(-1)
                
                if not hasattr(self, '_attn_dim_warning_shown'):
                    print(f"警告: GATLayer注意力维度不匹配。调整注意力向量从{self.a.size(1)}到{actual_dim}")
                    self._attn_dim_warning_shown = True
                    
                new_a = nn.Parameter(torch.zeros(size=(self.num_heads, actual_dim, 1), device=device))
                nn.init.xavier_uniform_(new_a)
                self.a = new_a
            
            # 计算注意力分数
            e = torch.matmul(all_combinations.view(batch_size, self.num_heads, N*N, -1), self.a)
            e = e.view(batch_size, self.num_heads, N, N)  # reshape回[batch, heads, nodes, nodes]
        
        # Apply LeakyReLU and masking
        e = self.leaky_relu(e)
        
        # Apply adjacency matrix mask (set attention to zero for non-connected nodes)
        # 使用FP16兼容的值（-1e4而不是-1e9，避免FP16溢出）
        adj = adj.unsqueeze(1).expand(batch_size, self.num_heads, N, N)
        e = e.masked_fill(adj == 0, float('-1e4'))
        
        # Apply softmax to get attention weights
        attention = F.softmax(e, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to get output features
        h_prime = torch.matmul(attention, Wh)  # [batch, heads, nodes, out_features]
        
        # 将多头注意力的结果平均而不是拼接，以保持维度一致
        # 原始实现 (拼接): h_prime = h_prime.transpose(1, 2).contiguous().view(batch_size, N, -1)
        # 新实现 (平均): 沿着head维度取平均值
        h_prime = h_prime.mean(dim=1)  # [batch, nodes, out_features]
        
        # 打印维度信息用于调试
        # print(f"GATLayer输出维度: {h_prime.shape}, 期望维度=[batch, nodes, {self.out_features}]")
        
        return h_prime


class TaskHeads(nn.Module):
    """Task-specific prediction heads for multi-task learning."""
    def __init__(self, input_dim=128):
        super(TaskHeads, self).__init__()
        
        # 增强版分类头，用于互作预测
        # 替换BatchNorm为LayerNorm，增加层复杂度
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 增强版检索头，用于嵌入学习
        # 使用硬编码维度，设定输入维度为32（实际CNN输出维度）
        self.human_encoder = nn.Sequential(
            nn.Linear(32, 256),  # 使用明确的32维输入
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        self.virus_encoder = nn.Sequential(
            nn.Linear(32, 256),  # 使用明确的32维输入
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        # 添加交互融合层，用于融合人类和病毒特征
        # 固定输入维度为当前使用的64维（32+32）
        self.cross_fusion = nn.Sequential(
            nn.Linear(64, input_dim),  # 64维输入（两个32维特征连接）
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128)
        )
        
        # 注意：motif预测已移除，未来将使用attention score解码motif
        
    def forward(self, global_features, sequence_features=None, attention_weights=None):
        """
        Forward pass through all task heads.
        
        Args:
            global_features: Global protein pair features for classification and retrieval
            sequence_features: Per-position features (保留用于未来motif解码，但不参与训练)
            attention_weights: Cross-chain attention weights from the model
                可以是单个张量 [batch, seq_q, seq_k] 或元组 (attn_h2v, attn_v2h)
                其中 attn_h2v 是 [batch, human_len, virus_len]，attn_v2h 是 [batch, virus_len, human_len]
                注意：attention_weights保留用于未来从attention score解码motif
                
        Returns:
            Dictionary of task outputs
        """
        batch_size = global_features.size(0)
        
        # 分割全局特征，获取人类和病毒蛋白质的特征
        feature_dim = global_features.size(1) // 3 if global_features.size(1) % 3 == 0 else global_features.size(1) // 2
        
        # 自适应处理不同维度情况
        if global_features.size(1) % 3 == 0:
            # 包含病毒家族特征
            human_features = global_features[:, :feature_dim]
            virus_features = global_features[:, feature_dim:2*feature_dim]
            family_features = global_features[:, 2*feature_dim:]
        else:
            # 不包含病毒家族特征
            human_features = global_features[:, :feature_dim]
            virus_features = global_features[:, feature_dim:]
            family_features = None
            
        # 任务1: 分类 - 预测互作概率
        try:
            interaction_prob = self.classification_head(global_features)
        except RuntimeError as e:
            print(f"分类头错误: {e}")
            print(f"全局特征形状: {global_features.shape}")
            print(f"分类头期望维度: {list(self.classification_head.parameters())[0].shape[1]}")
            # 尝试调整输入维度
            projection = nn.Linear(global_features.size(1), 
                                list(self.classification_head.parameters())[0].shape[1]).to(global_features.device)
            adjusted_features = projection(global_features)
            interaction_prob = self.classification_head(adjusted_features)
        
        # 任务2: 检索 - 计算相似度矩阵
        # 增强版嵌入表示学习
        try:
            # 直接将固定维度特征传入编码器
            human_embed = self.human_encoder(human_features)
            virus_embed = self.virus_encoder(virus_features)
            print(f"人类特征形状: {human_features.shape}, 病毒特征形状: {virus_features.shape}")
        except RuntimeError as e:
            print(f"编码器错误: {e}")
            print(f"人类特征形状: {human_features.shape}, 病毒特征形状: {virus_features.shape}")
            
            # 发生错误时创建符合期望尺寸的编码结果
            human_embed = torch.zeros(human_features.size(0), 128, device=human_features.device)
            virus_embed = torch.zeros(virus_features.size(0), 128, device=virus_features.device)
            nn.init.normal_(human_embed, 0, 0.1)
            nn.init.normal_(virus_embed, 0, 0.1)
        
        # 使用交互融合层得到融合特征（可选）
        fusion_embed = None
        if hasattr(self, 'cross_fusion'):
            # 将人类和病毒特征连接起来
            combined_features = torch.cat([human_features, virus_features], dim=1)
            print(f"combined_features 维度: {combined_features.shape}")
            
            try:
                # 直接将连接的特征传入融合层
                # 我们已经设计为接受64维输入（两个32维特征连接）
                fusion_embed = self.cross_fusion(combined_features)
            except RuntimeError as e:
                print(f"融合层错误: {e}")
                print(f"combined_features 维度: {combined_features.shape}")
                # 错误恢复机制
                fusion_embed = torch.zeros(combined_features.size(0), 128, device=combined_features.device)
                nn.init.normal_(fusion_embed, 0, 0.1)
        
        # L2规范化，用于余弦相似度计算
        human_embed_norm = F.normalize(human_embed, p=2, dim=1)
        virus_embed_norm = F.normalize(virus_embed, p=2, dim=1)
        
        # 计算相似度矩阵用于检索任务评估
        similarity_matrix = torch.mm(human_embed_norm, virus_embed_norm.transpose(0, 1))
        
        # 注意：motif预测已移除，未来将使用attention_weights解码motif
        # sequence_features和attention_weights保留在输出中，但不参与训练损失计算
        # 这些信息可以用于未来从attention score解码motif
        
        # 组合所有任务的输出
        outputs = {
            'interaction_prob': interaction_prob.squeeze(-1) if hasattr(interaction_prob, 'squeeze') else interaction_prob,  # [batch]
            'similarity_matrix': similarity_matrix,  # [batch, batch]
            'human_embed': human_embed,  # [batch, dim]
            'virus_embed': virus_embed,  # [batch, dim]
            'human_embedding': human_embed_norm,  # 兼容旧代码
            'virus_embedding': virus_embed_norm  # 兼容旧代码
        }
        
        # 添加融合嵌入(如果有)
        if fusion_embed is not None:
            outputs['fusion_embed'] = fusion_embed
        
        # 保留attention_weights用于未来motif解码（不参与训练）
        if attention_weights is not None:
            if isinstance(attention_weights, tuple):
                outputs['h2v_weights'] = attention_weights[0]
                outputs['v2h_weights'] = attention_weights[1]
            else:
                outputs['attention_weights'] = attention_weights
            
        return outputs
        # 组合所有任务的输出
        outputs = {
            'interaction_prob': interaction_prob.squeeze(-1) if hasattr(interaction_prob, 'squeeze') else interaction_prob,  # [batch]
            'similarity_matrix': similarity_matrix,  # [batch, batch]
            'human_embed': human_embed,  # [batch, dim]
            'virus_embed': virus_embed,  # [batch, dim]
            'human_embedding': human_embed_norm,  # 兼容旧代码
            'virus_embedding': virus_embed_norm  # 兼容旧代码
        }
        
        # 添加融合嵌入(如果有)
        if fusion_embed is not None:
            outputs['fusion_embed'] = fusion_embed
        
        if motif_scores is not None:
            outputs.update({'motif_scores': motif_scores})
            
        return outputs


class ViralFamilyEncoder(nn.Module):
    """
    Encodes viral family information as additional features.
    This helps the model learn family-specific interaction patterns.
    """
    def __init__(self, num_families=10, embedding_dim=32, output_dim=64):
        super(ViralFamilyEncoder, self).__init__()
        self.family_embedding = nn.Embedding(num_families + 1, embedding_dim)  # +1 for unknown
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, family_ids):
        """
        Forward pass to encode viral family information.
        
        Args:
            family_ids: Tensor of family IDs [batch_size]
            
        Returns:
            Family embeddings [batch_size, output_dim]
        """
        embedded = self.family_embedding(family_ids)
        return self.projection(embedded)


class SuperConnectorAttention(nn.Module):
    """
    Special attention mechanism for highly connected proteins.
    Scales attention based on protein connectivity.
    """
    def __init__(self, input_dim=64, max_connections=1000, dropout=0.1):
        super(SuperConnectorAttention, self).__init__()
        self.max_connections = max_connections
        self.feature_dim = input_dim  # 使用input_dim作为内部feature_dim
        
        # Feature transformation - 确保输出维度与输入特征维度匹配
        self.connection_transform = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, self.feature_dim)  # 使用self.feature_dim而不是feature_dim
        )
        
        # Attention scaling parameters
        self.scaling_factor = nn.Parameter(torch.tensor(1.0))
        self.min_scale = 0.5
        
    def forward(self, features, connectivity_counts):
        """
        Apply connectivity-aware attention scaling.
        
        Args:
            features: Protein features [batch_size, feature_dim]
            connectivity_counts: Number of interactions for each protein [batch_size]
            
        Returns:
            Scaled features with connectivity information
        """
        # 确保特征维度正确 
        actual_feature_dim = features.size(-1)
        
        # 如果输入特征维度与初始化时指定的维度不匹配，输出警告并调整连接特征变换层
        if actual_feature_dim != self.feature_dim and not hasattr(self, '_dim_warning_shown'):
            print(f"警告: SuperConnectorAttention期望特征维度{self.feature_dim}，但收到{actual_feature_dim}")
            # 标记已显示警告，避免重复输出
            self._dim_warning_shown = True
            
            # 动态调整连接变换层以匹配实际输入维度
            device = features.device
            self.connection_transform = nn.Sequential(
                nn.Linear(1, 16, device=device),
                nn.ReLU(),
                nn.Linear(16, actual_feature_dim, device=device)
            )
            # 更新存储的特征维度
            self.feature_dim = actual_feature_dim
        
        # Normalize connectivity counts
        norm_counts = (connectivity_counts.float() / self.max_connections).clamp(0, 1).unsqueeze(1)
        
        # Transform connectivity information 
        conn_features = self.connection_transform(norm_counts)
        
        # Calculate attention scaling factors (保持广播兼容性)
        scale = torch.sigmoid(norm_counts * self.scaling_factor) + self.min_scale
        
        # Scale features based on connectivity
        scaled_features = features * scale + conn_features
        
        return scaled_features
