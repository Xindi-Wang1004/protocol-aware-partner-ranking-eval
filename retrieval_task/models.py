"""
检索任务模型定义
包含5种配置：
1. ESM3微调 + 检索头
2. ESM3冻结 + 检索头
3. ESM3 + CNN + 检索头
4. ESM3 LoRA + 检索头
5. ESM3 LoRA + Batch内正样本图GNN + 检索头（方案A）
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

# 复用 base_modules 的 GATLayer（方案A batch 内图），按路径加载避免与 site-packages 的 models 冲突
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_base_modules_path = os.path.join(_project_root, "models", "base_modules.py")
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("base_modules", _base_modules_path)
    _base_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_base_mod)
    GATLayer = getattr(_base_mod, "GATLayer", None)
except Exception:
    GATLayer = None


class ESM3RetrievalModel(nn.Module):
    """ESM3 + 检索头模型"""
    def __init__(self, esm_model_name="esm3-open", freeze_esm=False, embed_dim=128, device=None, shared_esm_model=None):
        super().__init__()
        self.freeze_esm = freeze_esm
        self.embed_dim = embed_dim
        
        # 设置设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 内存优化：如果提供了共享的ESM3模型，直接使用，避免重复加载
        if shared_esm_model is not None:
            print(f"使用共享的ESM3模型实例（内存优化）")
            self.esm_model = shared_esm_model
        else:
            # 尝试从本地缓存加载ESM3模型
            self.esm_model = self._load_esm3_model(esm_model_name)
        
        # 冻结或解冻ESM3参数
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        # 获取ESM3的嵌入维度
        with torch.no_grad():
            test_protein = ESMProtein(sequence="ACDEFG")
            encoded = self.esm_model.encode(test_protein)
            results = self.esm_model.logits(encoded, config=LogitsConfig(return_embeddings=True))
            self.esm_dim = results.embeddings.shape[-1]
        
        # 投影到检索嵌入维度
        self.human_proj = nn.Linear(self.esm_dim, embed_dim)
        self.virus_proj = nn.Linear(self.esm_dim, embed_dim)
    
    def _load_esm3_model(self, esm_model_name):
        """加载ESM3模型，优先使用本地缓存。离线友好：先加载到 CPU 再 to(device)，避免多卡保存单卡加载失败。"""
        model_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")
        model_cache_path = os.path.join(model_cache_dir, f"{esm_model_name}.pt")
        os.makedirs(model_cache_dir, exist_ok=True)
        use_local_only = os.environ.get("ESM3_OFFLINE", os.environ.get("ESM3_LOCAL_ONLY", ""))

        if os.path.exists(model_cache_path):
            for map_loc, desc in [(torch.device("cpu"), "cpu"), (self.device, "device")]:
                try:
                    print(f"尝试从本地缓存加载ESM3模型: {model_cache_path} (map_location={desc})")
                    esm_model = torch.load(model_cache_path, map_location=map_loc, weights_only=False)
                    esm_model = esm_model.to(self.device)
                    print(f"ESM3模型从本地缓存加载成功! (设备: {self.device})")
                    return esm_model
                except Exception as e:
                    print(f"从本地缓存加载失败 [{desc}]: {e}")
                    if use_local_only:
                        raise RuntimeError(
                            f"ESM3 仅使用本地缓存（ESM3_OFFLINE/ESM3_LOCAL_ONLY 已设置），本地加载失败: {e}"
                        ) from e
            if use_local_only:
                raise RuntimeError(f"ESM3 仅使用本地缓存，但从 {model_cache_path} 加载失败。")
            print("将尝试从 Hugging Face 加载（若离线请设置 ESM3_OFFLINE=1 仅用本地）")
        elif use_local_only:
            raise FileNotFoundError(
                f"ESM3 仅使用本地缓存，但未找到: {model_cache_path}。请先在有网络环境下载并保存到该路径。"
            )

        print(f"从 Hugging Face 加载ESM3模型: {esm_model_name}")
        try:
            esm_model = ESM3.from_pretrained(esm_model_name).to(self.device)
            print(f"ESM3模型加载成功! (设备: {self.device})")
        except Exception as load_error:
            # 如果直接加载失败，检查是否是token问题
            error_str = str(load_error).lower()
            if "token" in error_str or "authentication" in error_str or "login" in error_str or "403" in error_str:
                # 需要token的情况
                from huggingface_hub import login
                token = None
                for env_var in ["HUGGINGFACE_TOKEN", "HF_TOKEN", "ESM3_HUGGINGFACE_TOKEN"]:
                    if env_var in os.environ:
                        token = os.environ[env_var]
                        print(f"使用环境变量 {env_var} 进行Hugging Face认证")
                        break
                
                if token:
                    try:
                        print("使用环境变量token进行Hugging Face自动登录")
                        login(token=token, add_to_git_credential=False)
                        print("Hugging Face自动登录成功!")
                        # 重新尝试加载
                        esm_model = ESM3.from_pretrained(esm_model_name).to(self.device)
                        print(f"ESM3模型加载成功! (设备: {self.device})")
                    except Exception as e:
                        print(f"使用环境变量token登录失败: {e}")
                        raise
                else:
                    print("\n================================================")
                    print("错误: 未找到Hugging Face认证token环境变量")
                    print("\n请设置环境变量:")
                    print("  export HUGGINGFACE_TOKEN='your_token'")
                    print("\n或者使用以下命令进行手动登录:")
                    print("  huggingface-cli login")
                    print("================================================\n")
                    raise RuntimeError("需要Hugging Face token才能加载ESM3模型")
            else:
                # 其他错误，直接抛出
                raise
        
        # 保存到本地缓存
        try:
            print(f"保存ESM3模型到本地缓存: {model_cache_path}")
            torch.save(esm_model, model_cache_path)
            print("模型缓存保存成功!")
        except Exception as save_err:
            print(f"保存模型缓存失败: {save_err}")
        
        return esm_model
    
    def forward(self, human_seqs, virus_seqs):
        """前向传播，返回人类和病毒的检索嵌入
        Args:
            human_seqs: List of human protein sequences
            virus_seqs: List of virus protein sequences
        """
        batch_size = len(human_seqs)
        human_embs = []
        virus_embs = []
        
        # 获取ESM3嵌入（逐个处理）
        with torch.set_grad_enabled(not self.freeze_esm):
            for i in range(batch_size):
                human_protein = ESMProtein(sequence=human_seqs[i])
                encoded_human = self.esm_model.encode(human_protein)
                human_results = self.esm_model.logits(
                    encoded_human,
                    config=LogitsConfig(return_embeddings=True)
                )
                human_emb = human_results.embeddings.mean(dim=1)  # 平均池化
                human_embs.append(human_emb)
                
                virus_protein = ESMProtein(sequence=virus_seqs[i])
                encoded_virus = self.esm_model.encode(virus_protein)
                virus_results = self.esm_model.logits(
                    encoded_virus,
                    config=LogitsConfig(return_embeddings=True)
                )
                virus_emb = virus_results.embeddings.mean(dim=1)  # 平均池化
                virus_embs.append(virus_emb)
        
        # 拼接批次
        human_emb = torch.cat(human_embs, dim=0)
        virus_emb = torch.cat(virus_embs, dim=0)
        
        # 确保嵌入在正确的设备上（ESM3生成的嵌入应该在GPU上，但为了安全起见显式移动）
        human_emb = human_emb.to(self.device)
        virus_emb = virus_emb.to(self.device)
        
        # 投影到检索嵌入空间
        human_embed = F.normalize(self.human_proj(human_emb), p=2, dim=-1)
        virus_embed = F.normalize(self.virus_proj(virus_emb), p=2, dim=-1)
        
        return human_embed, virus_embed


class ESM3CNNRetrievalModel(nn.Module):
    """ESM3 + CNN + 检索头模型"""
    def __init__(self, esm_model_name="esm3-open", freeze_esm=False, hidden_dim=256, embed_dim=128, device=None, shared_esm_model=None):
        super().__init__()
        self.freeze_esm = freeze_esm
        self.embed_dim = embed_dim
        
        # 设置设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 内存优化：如果提供了共享的ESM3模型，直接使用，避免重复加载
        if shared_esm_model is not None:
            print(f"使用共享的ESM3模型实例（内存优化）")
            self.esm_model = shared_esm_model
        else:
            # 尝试从本地缓存加载ESM3模型
            self.esm_model = self._load_esm3_model(esm_model_name)
        
        # 冻结或解冻ESM3参数
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        # 获取ESM3的嵌入维度
        with torch.no_grad():
            test_protein = ESMProtein(sequence="ACDEFG")
            encoded = self.esm_model.encode(test_protein)
            results = self.esm_model.logits(encoded, config=LogitsConfig(return_embeddings=True))
            self.esm_dim = results.embeddings.shape[-1]
        
        # CNN特征提取
        self.human_cnn = nn.Sequential(
            nn.Conv1d(self.esm_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.virus_cnn = nn.Sequential(
            nn.Conv1d(self.esm_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 投影到检索嵌入维度
        self.human_proj = nn.Linear(hidden_dim, embed_dim)
        self.virus_proj = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, human_seqs, virus_seqs):
        """前向传播，返回人类和病毒的检索嵌入
        Args:
            human_seqs: List of human protein sequences
            virus_seqs: List of virus protein sequences
        """
        batch_size = len(human_seqs)
        human_embs = []
        virus_embs = []
        
        # 获取ESM3嵌入（逐个处理）
        with torch.set_grad_enabled(not self.freeze_esm):
            for i in range(batch_size):
                human_protein = ESMProtein(sequence=human_seqs[i])
                encoded_human = self.esm_model.encode(human_protein)
                human_results = self.esm_model.logits(
                    encoded_human,
                    config=LogitsConfig(return_embeddings=True)
                )
                human_emb = human_results.embeddings  # [1, seq_len, esm_dim]
                human_embs.append(human_emb)
                
                virus_protein = ESMProtein(sequence=virus_seqs[i])
                encoded_virus = self.esm_model.encode(virus_protein)
                virus_results = self.esm_model.logits(
                    encoded_virus,
                    config=LogitsConfig(return_embeddings=True)
                )
                virus_emb = virus_results.embeddings  # [1, seq_len, esm_dim]
                virus_embs.append(virus_emb)
        
        # 拼接批次
        human_emb = torch.cat(human_embs, dim=0)  # [batch, seq_len, esm_dim]
        virus_emb = torch.cat(virus_embs, dim=0)  # [batch, seq_len, esm_dim]
        
        # 确保嵌入在正确的设备上
        human_emb = human_emb.to(self.device)
        virus_emb = virus_emb.to(self.device)
        
        # CNN特征提取
        human_emb = human_emb.transpose(1, 2)  # [batch, esm_dim, seq_len]
        human_features = self.human_cnn(human_emb).squeeze(-1)  # [batch, hidden_dim]
        
        virus_emb = virus_emb.transpose(1, 2)  # [batch, esm_dim, seq_len]
        virus_features = self.virus_cnn(virus_emb).squeeze(-1)  # [batch, hidden_dim]
        
        # 投影到检索嵌入空间
        human_embed = F.normalize(self.human_proj(human_features), p=2, dim=-1)
        virus_embed = F.normalize(self.virus_proj(virus_features), p=2, dim=-1)
        
        return human_embed, virus_embed
    
    def _load_esm3_model(self, esm_model_name):
        """加载ESM3模型，复用 ESM3RetrievalModel 的本地/离线加载逻辑"""
        temp = ESM3RetrievalModel(esm_model_name=esm_model_name, freeze_esm=True, device=self.device)
        return temp.esm_model


class ESM3LoRARetrievalModel(nn.Module):
    """ESM3 LoRA + 检索头模型
    使用 LoRA 技术，只训练少量参数，大幅减少显存占用
    """
    def __init__(self, esm_model_name="esm3-open", lora_rank=8, lora_alpha=16, 
                 lora_dropout=0.0, embed_dim=128, device=None, 
                 shared_esm_model=None, target_modules=None):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 设置设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 内存优化：如果提供了共享的ESM3模型，直接使用，避免重复加载
        if shared_esm_model is not None:
            print(f"使用共享的ESM3模型实例（内存优化）")
            self.esm_model = shared_esm_model
        else:
            # 尝试从本地缓存加载ESM3模型
            self.esm_model = self._load_esm3_model(esm_model_name)
        
        # 应用 LoRA 到 ESM3 模型
        # 默认应用到 transformer 层的注意力机制和 FFN
        if target_modules is None:
            # ESM3 的典型模块名称模式 - 尝试匹配常见的线性层
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                             'gate_proj', 'up_proj', 'down_proj', 
                             'fc1', 'fc2', 'attention', 'ffn', 'linear']
        
        # 导入 LoRA 模块（从classification_task中复用）
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from classification_task.models_lora import apply_lora_to_model
        
        print(f"应用 LoRA (rank={lora_rank}, alpha={lora_alpha}) 到 ESM3 模型...")
        self.esm_model = apply_lora_to_model(
            self.esm_model, 
            target_modules=target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
        
        # 统计可训练参数
        total_params = sum(p.numel() for p in self.esm_model.parameters())
        trainable_params = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
        print(f"ESM3 模型参数统计: 总参数={total_params:,}, 可训练参数={trainable_params:,} "
              f"({100*trainable_params/total_params:.2f}%)")
        
        # 获取ESM3的嵌入维度
        with torch.no_grad():
            test_protein = ESMProtein(sequence="ACDEFG")
            encoded = self.esm_model.encode(test_protein)
            results = self.esm_model.logits(encoded, config=LogitsConfig(return_embeddings=True))
            self.esm_dim = results.embeddings.shape[-1]
        
        # 投影到检索嵌入维度
        self.human_proj = nn.Linear(self.esm_dim, embed_dim)
        self.virus_proj = nn.Linear(self.esm_dim, embed_dim)
    
    def _load_esm3_model(self, esm_model_name):
        """加载ESM3模型，优先使用本地缓存（复用 ESM3RetrievalModel 的方法）"""
        # 创建一个临时实例来复用加载逻辑
        temp_model = ESM3RetrievalModel(
            esm_model_name=esm_model_name,
            freeze_esm=True,
            device=self.device
        )
        return temp_model.esm_model
    
    def forward(self, human_seqs, virus_seqs):
        """前向传播，返回人类和病毒的检索嵌入
        Args:
            human_seqs: List of human protein sequences
            virus_seqs: List of virus protein sequences
        """
        batch_size = len(human_seqs)
        human_embs = []
        virus_embs = []
        
        # 获取ESM3嵌入（逐个处理）
        # LoRA模式下，需要梯度用于训练LoRA参数
        for i in range(batch_size):
            human_protein = ESMProtein(sequence=human_seqs[i])
            encoded_human = self.esm_model.encode(human_protein)
            human_results = self.esm_model.logits(
                encoded_human,
                config=LogitsConfig(return_embeddings=True)
            )
            human_emb = human_results.embeddings.mean(dim=1)  # 平均池化
            human_embs.append(human_emb)
            
            virus_protein = ESMProtein(sequence=virus_seqs[i])
            encoded_virus = self.esm_model.encode(virus_protein)
            virus_results = self.esm_model.logits(
                encoded_virus,
                config=LogitsConfig(return_embeddings=True)
            )
            virus_emb = virus_results.embeddings.mean(dim=1)  # 平均池化
            virus_embs.append(virus_emb)
        
        # 拼接批次
        human_emb = torch.cat(human_embs, dim=0)
        virus_emb = torch.cat(virus_embs, dim=0)
        
        # 确保嵌入在正确的设备上
        human_emb = human_emb.to(self.device)
        virus_emb = virus_emb.to(self.device)
        
        # 投影到检索嵌入空间
        human_embed = F.normalize(self.human_proj(human_emb), p=2, dim=-1)
        virus_embed = F.normalize(self.virus_proj(virus_emb), p=2, dim=-1)
        
        return human_embed, virus_embed


class ESM3LoRAGNNRetrievalModel(nn.Module):
    """方案A：ESM3+LoRA + batch内正样本对建图 + GAT，检索头。
    ESM3 从本地 model_cache 加载，与 ESM3LoRARetrievalModel 一致，不清缓存。
    """
    def __init__(self, esm_model_name="esm3-open", lora_rank=8, lora_alpha=16,
                 lora_dropout=0.0, embed_dim=128, device=None, shared_esm_model=None,
                 target_modules=None, num_gnn_layers=2, gat_heads=4):
        super().__init__()
        if GATLayer is None:
            raise ImportError("方案A 需要 models.base_modules.GATLayer")
        self.embed_dim = embed_dim
        self.num_gnn_layers = num_gnn_layers
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if shared_esm_model is not None:
            self.esm_model = shared_esm_model
        else:
            self.esm_model = self._load_esm3_model(esm_model_name)

        if target_modules is None:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj',
                             'fc1', 'fc2', 'attention', 'ffn', 'linear']
        from classification_task.models_lora import apply_lora_to_model
        self.esm_model = apply_lora_to_model(
            self.esm_model, target_modules=target_modules,
            rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
        )
        self.freeze_esm = True
        with torch.no_grad():
            test_protein = ESMProtein(sequence="ACDEFG")
            encoded = self.esm_model.encode(test_protein)
            results = self.esm_model.logits(encoded, config=LogitsConfig(return_embeddings=True))
            esm_dim = results.embeddings.shape[-1]
        self.human_proj = nn.Linear(esm_dim, embed_dim)
        self.virus_proj = nn.Linear(esm_dim, embed_dim)

        in_dim = embed_dim
        self.gat_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gat_layers.append(GATLayer(in_dim, in_dim, num_heads=gat_heads))

    def _load_esm3_model(self, model_name):
        tmp = ESM3RetrievalModel(esm_model_name=model_name, device=self.device)
        return tmp.esm_model

    def forward(self, human_seqs, virus_seqs, labels=None):
        batch_size = len(human_seqs)
        human_embs, virus_embs = [], []
        # 与 ESM3LoRARetrievalModel 保持一致：encode + logits(return_embeddings)，再做平均池化
        with torch.set_grad_enabled(not self.freeze_esm):
            for i in range(batch_size):
                human_protein = ESMProtein(sequence=human_seqs[i])
                encoded_human = self.esm_model.encode(human_protein)
                human_results = self.esm_model.logits(
                    encoded_human,
                    config=LogitsConfig(return_embeddings=True)
                )
                human_emb = human_results.embeddings.mean(dim=1)
                human_embs.append(human_emb)

                virus_protein = ESMProtein(sequence=virus_seqs[i])
                encoded_virus = self.esm_model.encode(virus_protein)
                virus_results = self.esm_model.logits(
                    encoded_virus,
                    config=LogitsConfig(return_embeddings=True)
                )
                virus_emb = virus_results.embeddings.mean(dim=1)
                virus_embs.append(virus_emb)

        human_emb = torch.cat(human_embs, dim=0).to(self.device)
        virus_emb = torch.cat(virus_embs, dim=0).to(self.device)
        h_proj = self.human_proj(human_emb)
        v_proj = self.virus_proj(virus_emb)

        if labels is not None and self.num_gnn_layers > 0:
            # 节点 [B human, B virus] -> (1, 2B, embed_dim)
            x = torch.cat([h_proj, v_proj], dim=0).unsqueeze(0)
            # 正样本对 (i, B+i) 建边，并加自环
            B = batch_size
            adj = torch.zeros(1, 2 * B, 2 * B, device=x.device, dtype=x.dtype)
            for i in range(B):
                if labels[i].item() == 1:
                    adj[0, i, B + i] = 1.0
                    adj[0, B + i, i] = 1.0
            adj = adj + torch.eye(2 * B, device=x.device, dtype=x.dtype).unsqueeze(0)
            for gat in self.gat_layers:
                x = gat(x, adj) + x
            human_embed = F.normalize(x[0, :B], p=2, dim=-1)
            virus_embed = F.normalize(x[0, B:], p=2, dim=-1)
        else:
            human_embed = F.normalize(h_proj, p=2, dim=-1)
            virus_embed = F.normalize(v_proj, p=2, dim=-1)
        return human_embed, virus_embed


class ESM3LoRACNNRetrievalModel(nn.Module):
    """ESM3 LoRA + CNN + 检索头模型
    使用 LoRA 技术，只训练少量参数，大幅减少显存占用
    """
    def __init__(self, esm_model_name="esm3-open", lora_rank=8, lora_alpha=16, 
                 lora_dropout=0.0, hidden_dim=256, embed_dim=128, device=None, 
                 shared_esm_model=None, target_modules=None):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 设置设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 内存优化：如果提供了共享的ESM3模型，直接使用，避免重复加载
        if shared_esm_model is not None:
            print(f"使用共享的ESM3模型实例（内存优化）")
            self.esm_model = shared_esm_model
        else:
            # 尝试从本地缓存加载ESM3模型
            self.esm_model = self._load_esm3_model(esm_model_name)
        
        # 应用 LoRA 到 ESM3 模型
        # 默认应用到 transformer 层的注意力机制和 FFN
        if target_modules is None:
            # ESM3 的典型模块名称模式 - 尝试匹配常见的线性层
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                             'gate_proj', 'up_proj', 'down_proj', 
                             'fc1', 'fc2', 'attention', 'ffn', 'linear']
        
        # 导入 LoRA 模块（从classification_task中复用）
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from classification_task.models_lora import apply_lora_to_model
        
        print(f"应用 LoRA (rank={lora_rank}, alpha={lora_alpha}) 到 ESM3 模型...")
        self.esm_model = apply_lora_to_model(
            self.esm_model, 
            target_modules=target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
        
        # 统计可训练参数
        total_params = sum(p.numel() for p in self.esm_model.parameters())
        trainable_params = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
        print(f"ESM3 模型参数统计: 总参数={total_params:,}, 可训练参数={trainable_params:,} "
              f"({100*trainable_params/total_params:.2f}%)")
        
        # 获取ESM3的嵌入维度
        with torch.no_grad():
            test_protein = ESMProtein(sequence="ACDEFG")
            encoded = self.esm_model.encode(test_protein)
            results = self.esm_model.logits(encoded, config=LogitsConfig(return_embeddings=True))
            self.esm_dim = results.embeddings.shape[-1]
        
        # CNN特征提取
        self.human_cnn = nn.Sequential(
            nn.Conv1d(self.esm_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.virus_cnn = nn.Sequential(
            nn.Conv1d(self.esm_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 投影到检索嵌入维度
        self.human_proj = nn.Linear(hidden_dim, embed_dim)
        self.virus_proj = nn.Linear(hidden_dim, embed_dim)
    
    def _load_esm3_model(self, esm_model_name):
        """加载ESM3模型，优先使用本地缓存（复用 ESM3RetrievalModel 的方法）"""
        # 创建一个临时实例来复用加载逻辑
        temp_model = ESM3RetrievalModel(
            esm_model_name=esm_model_name,
            freeze_esm=True,
            device=self.device
        )
        return temp_model.esm_model
    
    def forward(self, human_seqs, virus_seqs):
        """前向传播，返回人类和病毒的检索嵌入
        Args:
            human_seqs: List of human protein sequences
            virus_seqs: List of virus protein sequences
        """
        batch_size = len(human_seqs)
        human_embs = []
        virus_embs = []
        
        # 获取ESM3嵌入（逐个处理）
        # LoRA模式下，需要梯度用于训练LoRA参数
        for i in range(batch_size):
            human_protein = ESMProtein(sequence=human_seqs[i])
            encoded_human = self.esm_model.encode(human_protein)
            human_results = self.esm_model.logits(
                encoded_human,
                config=LogitsConfig(return_embeddings=True)
            )
            human_emb = human_results.embeddings  # [1, seq_len, esm_dim]
            human_embs.append(human_emb)
            
            virus_protein = ESMProtein(sequence=virus_seqs[i])
            encoded_virus = self.esm_model.encode(virus_protein)
            virus_results = self.esm_model.logits(
                encoded_virus,
                config=LogitsConfig(return_embeddings=True)
            )
            virus_emb = virus_results.embeddings  # [1, seq_len, esm_dim]
            virus_embs.append(virus_emb)
        
        # 拼接批次
        human_emb = torch.cat(human_embs, dim=0)  # [batch, seq_len, esm_dim]
        virus_emb = torch.cat(virus_embs, dim=0)  # [batch, seq_len, esm_dim]
        
        # 确保嵌入在正确的设备上
        human_emb = human_emb.to(self.device)
        virus_emb = virus_emb.to(self.device)
        
        # CNN特征提取
        human_emb = human_emb.transpose(1, 2)  # [batch, esm_dim, seq_len]
        human_features = self.human_cnn(human_emb).squeeze(-1)  # [batch, hidden_dim]
        
        virus_emb = virus_emb.transpose(1, 2)  # [batch, esm_dim, seq_len]
        virus_features = self.virus_cnn(virus_emb).squeeze(-1)  # [batch, hidden_dim]
        
        # 投影到检索嵌入空间
        human_embed = F.normalize(self.human_proj(human_features), p=2, dim=-1)
        virus_embed = F.normalize(self.virus_proj(virus_features), p=2, dim=-1)
        
        return human_embed, virus_embed

