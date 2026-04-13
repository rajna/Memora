"""
Memora - Webpage-based Memory with PageRank
网页记忆系统 - 基于PageRank的重要性排序

Usage:
    from memory_system import Memora
    
    memora = Memora()
    node = memora.add_memory("Content...", tags=["project"])
    results = memora.search("query", top_k=5)

Backward Compatibility:
    MemorySystem is still available as an alias for Memora.
"""

from .memory_system import Memora, MemorySystem  # MemorySystem = 向后兼容别名
from .models import MemoryNode, SearchResult
from .storage import MemoryStorage
from .pagerank import MemoryGraph

__all__ = [
    'Memora',
    'MemorySystem',  # 向后兼容
    'MemoryNode',
    'SearchResult',
    'MemoryStorage',
    'MemoryGraph',
]

__version__ = '0.2.0'
