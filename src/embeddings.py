"""
Embedding Manager
向量嵌入管理器
使用 sentence-transformers 生成分布式表示
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from .config import EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_DIR
except ImportError:
    # 独立运行时的默认值
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    EMBEDDING_DIR = "data/embeddings"


class EmbeddingManager:
    """
    向量嵌入管理器
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_dir: str = EMBEDDING_DIR):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._dim = EMBEDDING_DIM
    
    def _get_model(self):
        """懒加载模型"""
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量数组 (N, dim)
        """
        model = self._get_model()
        # 清理文本
        cleaned = [t.strip() if t else "" for t in texts]
        embeddings = model.encode(cleaned, show_progress_bar=False)
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode([text])[0]
    
    def save_embedding(self, node_id: str, embedding: np.ndarray) -> str:
        """
        保存向量到文件
        
        Returns:
            文件路径
        """
        file_path = self.cache_dir / f"{node_id}.npy"
        np.save(file_path, embedding)
        return str(file_path)
    
    def load_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """加载向量"""
        file_path = self.cache_dir / f"{node_id}.npy"
        if file_path.exists():
            return np.load(file_path)
        return None
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray,
                          query_text: str = "", doc_text: str = "") -> float:
        """
        计算余弦相似度，支持关键词匹配增强
        
        Args:
            emb1: 查询向量
            emb2: 文档向量
            query_text: 查询文本（用于关键词匹配）
            doc_text: 文档文本（用于关键词匹配）
        
        Returns:
            相似度分数 [0, 1]
        """
        import re
        
        # 1. 计算语义相似度（基础）
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            semantic_score = 0.0
        else:
            cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
            semantic_score = (cosine_sim + 1) / 2
        
        # 2. 关键词匹配增强（如果提供了文本）
        if query_text and doc_text:
            query_lower = query_text.lower()
            doc_lower = doc_text.lower()
            
            keyword_score = 0.0
            
            # 精确短语匹配（最高权重）
            if query_lower in doc_lower:
                count = doc_lower.count(query_lower)
                keyword_score = min(0.8 + count * 0.05, 0.95)
            else:
                # 词项匹配
                # 提取英文单词（保留大小写变体）
                words = re.findall(r'[a-zA-Z]+', query_text)
                keywords = set()
                for w in words:
                    keywords.add(w.lower())
                    keywords.add(w.upper())
                    keywords.add(w)
                
                # 提取中文字符
                chinese = re.findall(r'[\u4e00-\u9fff]+', query_text)
                keywords.update(chinese)
                
                if keywords:
                    matches = sum(1 for kw in keywords if kw in doc_lower)
                    if matches > 0:
                        keyword_score = min(0.5 + matches * 0.1, 0.85)
            
            # 融合语义分和关键词分（关键词匹配强时提升整体分数）
            if keyword_score > semantic_score:
                # 关键词匹配好，大幅加权
                return max(keyword_score, semantic_score * 0.5 + keyword_score * 0.5)
            else:
                # 语义相似但关键词不明显，保持语义分
                return semantic_score
        
        return semantic_score
    
    def find_similar(
        self, 
        query_emb: np.ndarray, 
        candidates: List[tuple], 
        top_k: int = 10
    ) -> List[tuple]:
        """
        查找最相似的向量
        
        Args:
            query_emb: 查询向量
            candidates: [(node_id, embedding), ...]
            top_k: 返回前k个
            
        Returns:
            [(node_id, similarity), ...] 按相似度排序
        """
        similarities = []
        for node_id, emb in candidates:
            sim = self.compute_similarity(query_emb, emb)
            similarities.append((node_id, sim))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class TFIDFEmbeddingManager:
    """
    TF-IDF + jieba 中文分词 嵌入管理器
    无需sentence-transformers，纯统计方法，适合中文内容
    """
    
    def __init__(self, cache_dir: str = EMBEDDING_DIR, max_features: int = 5000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_features = max_features
        self._vectorizer = None
        self._fitted = False
        
        # 尝试导入jieba
        try:
            import jieba
            self.jieba = jieba
            self._use_jieba = True
        except ImportError:
            self._use_jieba = False
    
    def _tokenize(self, text: str) -> List[str]:
        """分词：中文用jieba，英文用空格"""
        import re
        
        tokens = []
        
        # 中文字符：用jieba分词
        chinese_parts = re.findall(r'[\u4e00-\u9fff]+', text)
        for part in chinese_parts:
            if self._use_jieba:
                tokens.extend(self.jieba.lcut(part))
            else:
                # fallback: 每个字作为一个token
                tokens.extend(list(part))
        
        # 英文单词
        english_parts = re.findall(r'[a-zA-Z]+', text.lower())
        tokens.extend(english_parts)
        
        # 数字
        numbers = re.findall(r'\d+', text)
        tokens.extend(numbers)
        
        return tokens
    
    def _preprocess(self, text: str) -> str:
        """预处理文本，返回空格分隔的token字符串（符合sklearn TF-IDF输入格式）"""
        tokens = self._tokenize(text)
        return ' '.join(tokens)
    
    def fit(self, texts: List[str]):
        """拟合TF-IDF向量器"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # 预处理所有文本
        processed = [self._preprocess(t) for t in texts]
        
        # 创建并拟合向量器
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            lowercase=False,  # 我们已经处理了大小写
            token_pattern=r'[^\s]+',  # 按空格分词（我们已经分好词了）
            min_df=1,  # 至少出现在1个文档中
            max_df=0.95,  # 忽略出现在95%以上文档中的词
        )
        self._vectorizer.fit(processed)
        self._fitted = True
        
        return self
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        编码文本为TF-IDF向量
        如果未fit过，会自动用这些文本fit
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # 预处理
        processed = [self._preprocess(t) for t in texts]
        
        if not self._fitted:
            # 自动fit
            n_docs = len(processed)
            # 动态调整 min_df/max_df，避免文档数过少时报错
            min_df_val = min(1, n_docs)  # 至少1个文档
            max_df_val = min(0.95, n_docs - 1) if n_docs > 1 else 1.0  # 确保不小于min_df
            max_df_val = max(0.5, max_df_val)  # 至少0.5
            
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                lowercase=False,
                token_pattern=r'[^\s]+',
                min_df=min_df_val,
                max_df=max_df_val,
            )
            vectors = self._vectorizer.fit_transform(processed)
            self._fitted = True
        else:
            vectors = self._vectorizer.transform(processed)
        
        return vectors.toarray()
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode([text])[0]
    
    def save_embedding(self, node_id: str, embedding: np.ndarray) -> str:
        """保存向量"""
        file_path = self.cache_dir / f"{node_id}.npy"
        np.save(file_path, embedding)
        return str(file_path)
    
    def load_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """加载向量"""
        file_path = self.cache_dir / f"{node_id}.npy"
        if file_path.exists():
            return np.load(file_path)
        return None
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray,
                          query_text: str = "", doc_text: str = "") -> float:
        """
        计算余弦相似度，支持关键词匹配增强
        """
        import re
        
        # 1. 计算TF-IDF余弦相似度（基础）
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            semantic_score = 0.0
        else:
            cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
            semantic_score = (cosine_sim + 1) / 2
        
        # 2. 关键词匹配增强
        if query_text and doc_text:
            query_lower = query_text.lower()
            doc_lower = doc_text.lower()
            
            keyword_score = 0.0
            
            # 精确短语匹配（最高权重）
            if query_lower in doc_lower:
                count = doc_lower.count(query_lower)
                keyword_score = min(0.8 + count * 0.05, 0.95)
            else:
                # 提取中文字符
                chinese = re.findall(r'[\u4e00-\u9fff]+', query_text)
                
                # 用jieba分词提取中文关键词
                if self._use_jieba and chinese:
                    keywords = set()
                    for part in chinese:
                        keywords.update(self.jieba.lcut(part))
                else:
                    keywords = set(chinese)
                
                # 提取英文单词
                words = re.findall(r'[a-zA-Z]+', query_text)
                keywords.update(w.lower() for w in words)
                
                if keywords:
                    matches = sum(1 for kw in keywords if kw in doc_lower)
                    if matches > 0:
                        keyword_score = min(0.5 + matches * 0.1, 0.85)
            
            # 融合分数
            if keyword_score > semantic_score:
                return max(keyword_score, semantic_score * 0.5 + keyword_score * 0.5)
        
        return semantic_score
    
    def find_similar(
        self,
        query_emb: np.ndarray,
        candidates: List[tuple],
        top_k: int = 10
    ) -> List[tuple]:
        """
        查找最相似的向量
        
        Args:
            query_emb: 查询向量
            candidates: [(node_id, embedding), ...]
            top_k: 返回前k个
            
        Returns:
            [(node_id, similarity), ...] 按相似度排序
        """
        similarities = []
        for node_id, emb in candidates:
            sim = self.compute_similarity(query_emb, emb)
            similarities.append((node_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class SimpleEmbeddingManager:
    """
    极简版嵌入管理器（无外部依赖）
    使用哈希技巧，作为最后的fallback
    """
    
    def __init__(self, dim: int = 384, cache_dir: str = EMBEDDING_DIR):
        self.dim = dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text.lower())
        return tokens
    
    def encode_single(self, text: str) -> np.ndarray:
        """哈希编码"""
        tokens = self._tokenize(text)
        vec = np.zeros(self.dim)
        for token in tokens:
            idx = hash(token) % self.dim
            vec[idx] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array([self.encode_single(t) for t in texts])
    
    def save_embedding(self, node_id: str, embedding: np.ndarray) -> str:
        file_path = self.cache_dir / f"{node_id}.npy"
        np.save(file_path, embedding)
        return str(file_path)
    
    def load_embedding(self, node_id: str) -> Optional[np.ndarray]:
        file_path = self.cache_dir / f"{node_id}.npy"
        return np.load(file_path) if file_path.exists() else None
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray,
                          query_text: str = "", doc_text: str = "") -> float:
        if query_text and doc_text:
            query_lower = query_text.lower()
            doc_lower = doc_text.lower()
            if query_lower in doc_lower:
                return min(0.7 + doc_lower.count(query_lower) * 0.05, 0.95)
        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return (np.dot(emb1, emb2) / (norm1 * norm2) + 1) / 2


def get_embedding_manager(backend: str = "auto"):
    """
    获取嵌入管理器
    
    Args:
        backend: "auto" | "sentence_transformers" | "tfidf" | "simple"
            - auto: 优先使用 sentence-transformers
            - sentence_transformers: 神经网络语义向量
            - tfidf: TF-IDF + jieba 中文分词
            - simple: 哈希技巧（无依赖）
    """
    if backend == "sentence_transformers":
        return EmbeddingManager()
    elif backend == "tfidf":
        return TFIDFEmbeddingManager()
    elif backend == "simple":
        return SimpleEmbeddingManager()
    elif backend == "auto":
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            return EmbeddingManager()
        else:
            try:
                import sklearn
                return TFIDFEmbeddingManager()
            except ImportError:
                return SimpleEmbeddingManager()
    else:
        raise ValueError(f"Unknown backend: {backend}")
