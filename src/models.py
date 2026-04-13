"""
Memory Node Model
记忆节点数据模型
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import hashlib
import json


@dataclass
class MemoryNode:
    """
    记忆节点 - 对应一个"网页"
    """
    id: str                          # 唯一ID (timestamp + hash)
    url: str                         # 虚拟URL (/memory/2026/04/08/abc123)
    created: datetime                # 创建时间
    modified: datetime               # 修改时间
    content: str                     # Markdown内容
    title: Optional[str] = None      # 标题
    
    # PageRank相关
    pagerank: float = 0.01           # PageRank分数（默认低值，新节点需通过backlinks提升）
    links: List[str] = field(default_factory=list)  # 外链URLs
    backlinks: List[str] = field(default_factory=list)  # 反链URLs
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    embedding_file: Optional[str] = None  # 向量文件路径
    
    # 来源信息
    source: Optional[str] = None     # 来源 (cli/telegram/etc)
    session_id: Optional[str] = None # 会话ID
    
    # 扩展字段
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "url": self.url,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "content": self.content,
            "title": self.title,
            "pagerank": self.pagerank,
            "links": self.links,
            "backlinks": self.backlinks,
            "tags": self.tags,
            "embedding_file": self.embedding_file,
            "source": self.source,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryNode":
        """从字典创建"""
        return cls(
            id=data["id"],
            url=data["url"],
            created=datetime.fromisoformat(data["created"]),
            modified=datetime.fromisoformat(data["modified"]),
            content=data["content"],
            title=data.get("title"),
            pagerank=data.get("pagerank", 0.01),
            links=data.get("links", []),
            backlinks=data.get("backlinks", []),
            tags=data.get("tags", []),
            embedding_file=data.get("embedding_file"),
            source=data.get("source"),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )
    
    def generate_id(self) -> str:
        """生成基于内容的唯一ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        timestamp = self.created.strftime("%Y%m%d%H%M")
        return f"{timestamp}-{content_hash}"
    
    def generate_url(self) -> str:
        """生成虚拟URL"""
        date_path = self.created.strftime("%Y/%m/%d")
        short_hash = self.id.split("-")[-1] if "-" in self.id else self.id[:8]
        return f"/memory/{date_path}/{short_hash}"


@dataclass
class SearchResult:
    """
    搜索结果
    """
    node: MemoryNode
    semantic_score: float      # 语义相似度
    pagerank_score: float      # PageRank分数
    recency_score: float       # 时效性分数
    final_score: float         # 最终加权分数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据（如扩展来源）
    
    def to_dict(self) -> Dict:
        return {
            "node": self.node.to_dict(),
            "semantic_score": self.semantic_score,
            "pagerank_score": self.pagerank_score,
            "recency_score": self.recency_score,
            "final_score": self.final_score,
            "metadata": self.metadata,
        }
