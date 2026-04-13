"""
Memory System Configuration
网页记忆系统配置
"""
import os

# Base paths
WORKSPACE = "/Users/rama/.nanobot/workspace"
MEMORY_DIR = os.path.join(WORKSPACE, "Memora", "data")
EMBEDDING_DIR = os.path.join(WORKSPACE, "Memora", "embeddings")
INDEX_DIR = os.path.join(WORKSPACE, "Memora", "index")

# PageRank settings
PAGERANK_DAMPING = 0.85  # 阻尼系数
PAGERANK_ITERATIONS = 100  # 迭代次数
PAGERANK_TOLERANCE = 1e-6  # 收敛阈值

# Retrieval weights
WEIGHT_SEMANTIC = 0.7    # 语义权重：最重要
WEIGHT_PAGERANK = 0.15   # PageRank权重：降低PR影响，更依赖语义匹配
WEIGHT_RECENCY = 0.0     # 时效性权重：不再作为正向加分（改为遗忘惩罚）

# Similarity thresholds
SIMILARITY_THRESHOLD_HIGH = 0.8  # 高相似度，自动建立链接
SIMILARITY_THRESHOLD_LOW = 0.3   # 低相似度阈值（短查询需要更宽松的阈值）

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 英文模型，384维
# EMBEDDING_MODEL = "BAAI/bge-small-zh"  # 中文优化模型，384维
# EMBEDDING_MODEL = "BAAI/bge-m3"  # 多语言模型，1024维
EMBEDDING_DIM = 384

# Recency decay (days)
RECENCY_HALF_LIFE = 30  # 30天半衰期
PAGERANK_IMPORTANCE_THRESHOLD = 0.002  # 核心知识阈值：超过此值的节点不受时间影响（基于PageRank均值0.0028）

# File patterns
MEMORY_FILE_PATTERN = "**/*.md"
