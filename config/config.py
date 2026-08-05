"""
Configuration settings for the protein interaction prediction model
"""

class ModelConfig:
    """Model architecture configuration"""
    # ESM embedding
    esm_model_name = "esm2_t33_650M_UR50D"  # Use smaller model for faster processing
    esm_embed_dim = 1280
    
    # CNN parameters
    cnn_out_dim = 64
    
    # Cross-attention parameters
    num_heads = 8
    dropout = 0.1
    
    # Graph neural network parameters
    graph_hidden_dim = 128
    num_gat_layers = 2
    attention_threshold = 0.1
    
    # Task-specific head parameters
    classifier_hidden_dim = 128
    retrieval_embed_dim = 128
    
    # New model parameters for enhanced model
    connectivity_embedding_dim = 64   # Embedding dimension for connectivity values
    viral_family_embedding_dim = 64   # Embedding dimension for viral families
    num_viral_families = 50           # Maximum number of viral families


class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 2e-5
    weight_decay = 0.01
    
    # Loss function weights
    alpha = 3.0  # Classification loss weight
    beta = 2.0   # Retrieval loss weight
    # 注意：motif预测已移除，未来将使用attention score解码motif
    
    # Learning rate scheduler
    warmup_steps = 100
    cosine_cycle_length = 5
    
    # Gradient accumulation steps (for effectively larger batch size)
    gradient_accumulation_steps = 4
    
    # Evaluation
    eval_steps = 500
    
    # Checkpointing
    save_steps = 1000
    checkpoint_dir = "checkpoints"
    
    # Hard negative mining
    neg_samples = 5
    margin = 0.3
    
    
class DataConfig:
    """Data configuration"""
    # Data paths
    data_path = "data/processed/protein_pairs.pkl"
    
    # Preprocessing
    # 根据训练集统计：人类序列99%分位数=2602，病毒序列99%分位数=3391
    # 设置为768可以覆盖约85%的序列，平衡内存使用和序列覆盖率，支持batch_size=2的检索任务训练
    max_length = 768  # 平衡序列覆盖率和内存使用，支持batch_size=2和检索任务训练
    
    # Splits
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    
    # Precomputed embeddings (to save time)
    use_precomputed_embeddings = False
    embedding_dir = "data/embeddings"
    
    # Dataset generation
    num_positive_samples = 20000
    num_negative_samples = 20000
    
    
class VisualizationConfig:
    """Visualization configuration"""
    # Attention visualization
    attention_cmap = "viridis"
    attention_figsize = (12, 10)
    
    # Motif visualization
    motif_cmap = "YlOrRd"
    motif_figsize = (15, 5)
    
    # Save directory
    save_dir = "visualizations"
    
    
# Create a unified config
class Config:
    """Complete configuration"""
    model = ModelConfig
    training = TrainingConfig
    data = DataConfig
    visualization = VisualizationConfig
    
    # Device configuration
    device = "cuda"  # Change to "cpu" if no GPU available
    
    # Logging
    use_wandb = False
    project_name = "protein-interaction"
    
    # Inference
    threshold = 0.5
    
    # Seed for reproducibility
    seed = 42
    
    # Constants for connectivity classification
    connectivity_thresholds = {
        'low': 1,          # 1-5 interactions
        'medium': 5,       # 6-20 interactions
        'high': 20,        # 21-100 interactions
        'super': 100       # >100 interactions ("super-connectors")
    }
    
    # Mapping of viral families to IDs
    viral_family_map = {
        'Retroviridae': 1,
        'Orthomyxoviridae': 2, 
        'Herpesviridae': 3,
        'Coronaviridae': 4,
        'Flaviviridae': 5,
        'Pneumoviridae': 6,
        'Rhabdoviridae': 7,
        'Picornaviridae': 8,
        'Papillomaviridae': 9,
        'Poxviridae': 10,
        'Filoviridae': 11,
        'Arenaviridae': 12,
        'Togaviridae': 13,
        'Paramyxoviridae': 14,
        'Polyomaviridae': 15,
        'Adenoviridae': 16,
        'Bunyaviridae': 17,
        'Hepadnaviridae': 18,
        'Caliciviridae': 19,
        'Parvoviridae': 20,
        'Unknown': 0
    }
    
    # Define task names for consistent reference
    tasks = {
        'classification': 'interaction_prediction',
        'retrieval': 'protein_retrieval',
        'motif': 'motif_prediction'
    }
    
    # Default model settings for different scales
    model_scales = {
        'small': {
            'embedding_dim': 768,
            'hidden_dim': 256,
            'num_heads': 4,
            'num_layers': 2,
            'batch_size': 64
        },
        'medium': {
            'embedding_dim': 1280,
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 4,
            'batch_size': 32
        },
        'large': {
            'embedding_dim': 1280,
            'hidden_dim': 1024,
            'num_heads': 16,
            'num_layers': 6,
            'batch_size': 16
        }
    }
