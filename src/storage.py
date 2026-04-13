"""
Storage Backend - Pure Markdown with YAML Frontmatter
存储后端 - 纯Markdown + YAML Frontmatter
"""
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Iterator
import frontmatter

from .models import MemoryNode


class MemoryStorage:
    """
    记忆存储管理器
    每个记忆节点 = 一个 .md 文件
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, node: MemoryNode) -> Path:
        """根据节点生成文件路径"""
        # 按日期组织: 2026/04/08/1730-a1b2c3d4.md
        date_folder = node.created.strftime("%Y/%m/%d")
        folder = self.base_dir / date_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{node.id}.md"
    
    def _url_to_path(self, url: str) -> Optional[Path]:
        """URL转文件路径"""
        # /memory/2026/04/08/abc123 -> 2026/04/08/*-abc123.md
        match = re.match(r"/memory/(\d{4}/\d{2}/\d{2})/(\w+)", url)
        if not match:
            return None
        date_path, short_hash = match.groups()
        folder = self.base_dir / date_path
        if not folder.exists():
            return None
        # 查找匹配的文件
        for f in folder.glob("*.md"):
            if short_hash in f.stem:
                return f
        return None
    
    def _is_valid_link(self, url: str) -> bool:
        """
        验证链接是否有效
        保留 Memora 的链接，过滤掉纯节点ID字符串
        """
        if not url:
            return False
        
        # Memory System 的 URL 格式: /memory/2026/04/10/982c8d77
        # 这些是有效的内部链接，不应该被过滤
        if url.startswith("/memory/"):
            return True
        
        # 提取路径最后部分
        parts = url.strip('/').split('/')
        if not parts:
            return False
        
        last_part = parts[-1]
        
        # 如果是8位十六进制（纯节点ID，没有/memory前缀），可能是误识别的，跳过
        if len(last_part) == 8 and all(c in '0123456789abcdef' for c in last_part.lower()):
            return False
        
        return True
    
    def _clean_links(self, node: MemoryNode) -> MemoryNode:
        """
        清理节点中的无效链接
        """
        # 过滤 links
        original_count = len(node.links)
        node.links = [url for url in node.links if self._is_valid_link(url)]
        filtered_count = original_count - len(node.links)
        
        if filtered_count > 0:
            print(f"  [Storage] 过滤 {filtered_count} 个疑似节点ID链接")
        
        return node

    def save(self, node: MemoryNode) -> str:
        """
        保存记忆节点到文件
        返回文件路径
        """
        # 如果没有ID，生成一个
        if not node.id:
            node.id = node.generate_id()
        if not node.url:
            node.url = node.generate_url()
        
        # 清理无效链接
        node = self._clean_links(node)
        
        file_path = self._get_file_path(node)
        
        # 构建YAML Frontmatter
        metadata = {
            "id": node.id,
            "url": node.url,
            "created": node.created.isoformat(),
            "modified": datetime.now().isoformat(),
            "title": node.title,
            "pagerank": node.pagerank,
            "links": node.links,
            "backlinks": node.backlinks,
            "tags": node.tags,
            "embedding_file": node.embedding_file,
            "source": node.source,
            "session_id": node.session_id,
        }
        # 过滤None值
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # 写入文件
        post = frontmatter.Post(node.content, **metadata)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
        
        return str(file_path)
    
    def load(self, file_path: str) -> Optional[MemoryNode]:
        """从文件加载记忆节点"""
        path = Path(file_path)
        if not path.exists():
            return None
        
        with open(path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        
        # 从frontmatter构建节点
        metadata_keys = ["id", "url", "created", "timestamp", "modified", "pagerank", "links", 
                         "backlinks", "tags", "embedding_file", "source", "session_id", "title"]
        extra_metadata = {k: post[k] for k in post.keys() if k not in metadata_keys and k != 'content'}
        
        # 支持 timestamp 或 created 字段
        created_value = post.get("created") or post.get("timestamp")
        if isinstance(created_value, datetime):
            created_dt = created_value
        elif isinstance(created_value, str):
            created_dt = datetime.fromisoformat(created_value)
        else:
            created_dt = datetime.now()
        
        # 处理 modified 字段（防止 None 值导致 fromisoformat 失败）
        modified_value = post.get("modified")
        if isinstance(modified_value, datetime):
            modified_dt = modified_value
        elif isinstance(modified_value, str):
            modified_dt = datetime.fromisoformat(modified_value)
        else:
            modified_dt = datetime.now()
        
        node = MemoryNode(
            id=post.get("id", path.stem),
            url=post.get("url", ""),
            created=created_dt,
            modified=modified_dt,
            content=post.content,
            title=post.get("title"),
            pagerank=post.get("pagerank", 1.0),
            links=post.get("links", []),
            backlinks=post.get("backlinks", []),
            tags=post.get("tags", []),
            embedding_file=post.get("embedding_file"),
            source=post.get("source"),
            session_id=post.get("session_id"),
            metadata=extra_metadata,
        )
        return node
    
    def load_by_url(self, url: str) -> Optional[MemoryNode]:
        """通过URL加载节点"""
        file_path = self._url_to_path(url)
        if file_path:
            return self.load(str(file_path))
        return None
    
    def load_by_id(self, node_id: str) -> Optional[MemoryNode]:
        """通过ID加载节点"""
        # ID格式: 202604081730-a1b2c3d4
        if len(node_id) >= 12:
            year = node_id[0:4]
            month = node_id[4:6]
            day = node_id[6:8]
            pattern = f"**/{year}/{month}/{day}/{node_id}.md"
        else:
            pattern = f"**/{node_id}.md"
        
        for file_path in self.base_dir.glob(pattern):
            return self.load(str(file_path))
        return None
    
    def iterate_all(self) -> Iterator[MemoryNode]:
        """遍历所有记忆节点"""
        for file_path in self.base_dir.rglob("*.md"):
            node = self.load(str(file_path))
            if node:
                yield node
    
    def get_all(self) -> List[MemoryNode]:
        """获取所有记忆节点"""
        return list(self.iterate_all())
    
    def delete(self, node_id: str) -> bool:
        """删除记忆节点"""
        node = self.load_by_id(node_id)
        if not node:
            return False
        file_path = self._get_file_path(node)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def update_backlinks(self, url: str, new_backlinks: List[str]):
        """更新节点的反链列表"""
        node = self.load_by_url(url)
        if node:
            node.backlinks = new_backlinks
            node.modified = datetime.now()
            self.save(node)
