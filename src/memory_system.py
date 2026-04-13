# -*- coding: utf-8 -*-
"""
Memory System - Main Entry Point
网页记忆系统 - 主入口

Core Concept:
- 将n-round对话转换为"网页"节点
- 每个节点有URL、内容、向量嵌入、PageRank分数
- 检索使用混合算法：语义相似度 + PageRank + 时效性
"""
import os
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from .config import MEMORY_DIR, EMBEDDING_DIR, INDEX_DIR
from .models import MemoryNode, SearchResult
from .storage import MemoryStorage
from .embeddings import get_embedding_manager, EmbeddingManager
from .pagerank import MemoryGraph, build_and_rank
from .retrieval import MemoryRetrieval, TwoStageRetriever


class Memora:
    """
    网页记忆系统主类
    
    Usage:
        ms = MemorySystem()
        
        # 添加记忆
        node = ms.add_memory("用户说...", tags=["project", "vrm"])
        
        # 从对话消息添加（自动检测 skills）
        node = ms.add_memory_from_messages(messages, source="auto-save")
        
        # 搜索
        results = ms.search("VRM动捕", top_k=5)
        
        # 获取相关记忆
        related = ms.get_related(node.url)
    """
    
    # Skill 检测相关配置
    SKILL_DIRS = [
        Path("/Users/rama/Documents/agi_nanobot/nanobot/nanobot/skills"),
        Path("/Users/rama/.nanobot/workspace/skills"),
    ]
    
    def __init__(
        self,
        memory_dir: str = MEMORY_DIR,
        embedding_dir: str = EMBEDDING_DIR,
        use_two_stage: bool = True,
        embedding_backend: str = "auto"
    ):
        """
        初始化记忆系统
        
        Args:
            memory_dir: 记忆存储目录
            embedding_dir: 向量嵌入缓存目录
            use_two_stage: 是否使用两阶段检索（TF-IDF粗排+语义精排）
            embedding_backend: 嵌入后端 ("auto", "sentence_transformers", "tfidf", "simple")
        """
        # 初始化目录
        self.memory_dir = memory_dir
        self.embedding_dir = embedding_dir
        os.makedirs(memory_dir, exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)
        
        # 初始化组件
        self.storage = MemoryStorage(memory_dir)
        self.embedding_mgr = get_embedding_manager(embedding_backend)
        
        # 选择检索策略
        if use_two_stage:
            self.retrieval = TwoStageRetriever(
                self.storage, 
                self.embedding_mgr,
                first_stage_k=100
            )
            print(f"Using Two-Stage Retrieval (TF-IDF + sentence-transformers)")
        else:
            self.retrieval = MemoryRetrieval(self.storage, self.embedding_mgr)
            print(f"Using Standard Retrieval")
        
        self.graph = MemoryGraph()
        
        print(f"Memora initialized")
        print(f"  Memory dir: {memory_dir}")
        print(f"  Embedding dir: {embedding_dir}")
        print(f"  Embedding backend: {type(self.embedding_mgr).__name__}")
        
        # Skill 名称缓存
        self._skills_cache = None
        self._skills_cache_time = 0
        self._skills_cache_ttl = 60  # 缓存60秒
    
    def _extract_tags_from_text(self, text: str, max_tags: int = 8) -> List[str]:
        """
        从文本内容提取标签（使用 jieba TextRank）
        
        Args:
            text: 文本内容
            max_tags: 最大标签数量
            
        Returns:
            标签列表
        """
        if not text or not text.strip():
            return []
        
        # 预处理：移除特殊标记
        text = re.sub(r'\[用户\]|\[AI\]|\[使用的技能\]', ' ', text)
        text = re.sub(r'```[\s\S]*?```', ' ', text)
        text = re.sub(r'`[^`]*`', ' ', text)
        text = re.sub(r'https?://\S+', ' ', text)
        
        # 尝试使用 jieba 提取关键词
        try:
            import jieba.analyse
            
            # TextRank 提取关键词
            keywords = jieba.analyse.textrank(
                text,
                topK=max_tags * 2,
                withWeight=False,
                allowPOS=('n', 'v', 'vn', 'nz', 'eng')  # 名词、动词、动名词、专有名词、英文
            )
            
            # 过滤停用词和太短的词
            stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '上', '也', 
                        '很', '到', '说', '要', '去', '你', '会', '着', '看', '好', '这', '那', '个',
                        '用户', 'ai', 'assistant', 'content', 'role', 'message'}
            
            tags = []
            for word in keywords:
                word = word.strip().lower()
                if (len(word) >= 2 and 
                    word not in stopwords and
                    not word.isdigit() and
                    word not in tags):
                    tags.append(word)
                
                if len(tags) >= max_tags:
                    break
            
            return tags
            
        except ImportError:
            # 如果没有 jieba，使用简单的词频统计
            words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text)
            from collections import Counter
            word_counts = Counter(w.lower() for w in words)
            return [w for w, _ in word_counts.most_common(max_tags) if len(w) >= 2]
        except Exception:
            return []
    
    def _get_skill_names(self) -> List[str]:
        """
        从 skill 目录获取实际存在的 skill 名称列表（带缓存）
        """
        import time
        now = time.time()
        if self._skills_cache is not None and (now - self._skills_cache_time) < self._skills_cache_ttl:
            return self._skills_cache
        
        skill_names = set()
        for skill_dir in self.SKILL_DIRS:
            if not skill_dir.exists():
                continue
            try:
                for item in skill_dir.iterdir():
                    if item.is_dir() and (item / "SKILL.md").exists():
                        name = item.name
                        if not name.startswith('.') and name not in ['__pycache__', 'node_modules']:
                            skill_names.add(name)
            except Exception:
                pass
        
        self._skills_cache = sorted(list(skill_names))
        self._skills_cache_time = now
        return self._skills_cache
    
    # 工具名到技能名的映射（处理工具名和技能目录名不一致的情况）
    TOOL_TO_SKILL_MAP = {
        'web_fetch': 'web-search',      # web_fetch 工具对应 web-search 技能
        'web_search': 'web-search',     # web_search 工具对应 web-search 技能
    }
    
    # 完全排除的工具（其 content 不应参与 skill 检测）
    # list_dir 输出的是目录列表，会误匹配 skill 名称
    EXCLUDED_TOOLS = {'list_dir'}
    
    def detect_skills_in_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        检测对话中使用了哪些 skill
        
        检测方法：
        1. 只从 role=tool 的消息中检测
        2. 根据 name 字段识别工具类型
        3. 排除 list_dir 的输出（因为是目录列表，会误匹配）
        4. 通过 TOOL_TO_SKILL_MAP 将工具名映射到技能名
        5. 对于没有映射的工具，从 content 中匹配 skill 名称
        
        Args:
            messages: 对话消息列表，每个消息是 dict，包含 role、name、content
            
        Returns:
            检测到的 skill 名称列表
        """
        skill_names = self._get_skill_names()
        if not skill_names:
            return []
        
        detected = set()
        
        for msg in messages:
            role = msg.get('role', '')
            name = msg.get('name', '')
            content = msg.get('content', '')
            
            if role != 'tool' or not name:
                continue
            
            name_lower = name.lower()
            
            # 跳过完全排除的工具（如 list_dir）
            if name_lower in self.EXCLUDED_TOOLS:
                continue
            
            # 检查是否有直接映射（web_search -> web-search）
            if name_lower in self.TOOL_TO_SKILL_MAP:
                skill_name = self.TOOL_TO_SKILL_MAP[name_lower]
                if skill_name and skill_name in skill_names:
                    detected.add(skill_name)
                continue
            
            # 没有映射的（如 read_file, exec），尝试从 content 中匹配 skill 名称
            if not content:
                continue
            
            # 跳过错误信息（避免误匹配错误消息中的路径）
            if content.startswith('Error:') or 'File not found' in content:
                continue
                
            content_lower = content.lower()
            for skill_name in skill_names:
                skill_lower = skill_name.lower()
                
                # 完全匹配
                if skill_lower == content_lower:
                    detected.add(skill_name)
                    break
                # 前缀匹配（如 agent-browser-0.2.0 匹配 agent-browser）
                elif skill_lower.startswith(content_lower + '-') or content_lower.startswith(skill_lower + '-'):
                    detected.add(skill_name)
                    break
                # 包含匹配
                elif skill_lower in content_lower or content_lower in skill_lower:
                    detected.add(skill_name)
                    break
        
        return sorted(list(detected))
    
    def format_conversation(self, messages: List[Dict[str, Any]], detected_skills: Optional[List[str]] = None) -> str:
        """
        将消息列表格式化为标准对话文本
        
        Args:
            messages: 对话消息列表
            detected_skills: 可选，已检测到的 skills，会追加到内容中
            
        Returns:
            格式化后的对话文本
        """
        parts = []
        last_role = None
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            # 严格过滤空内容（包括只有空白字符的）
            if not content or not content.strip():
                continue
            
            # 清理内容（移除首尾空白）
            content = content.strip()
                
            if role == 'user':
                parts.append(f"[用户] {content}")
                last_role = 'user'
            elif role == 'assistant':
                # 如果是连续的 AI 消息，合并到上一条
                if last_role == 'assistant' and parts:
                    parts[-1] += f"\n{content}"
                else:
                    parts.append(f"[AI] {content}")
                    last_role = 'assistant'
        
        # 如果有检测到的 skills，追加到最后一条 assistant 消息后面
        if detected_skills:
            skills_str = ', '.join(detected_skills)
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].startswith('[AI]'):
                    parts[i] += f"\n[使用的技能] {skills_str}"
                    break
        
        return "\n\n".join(parts)
    
    def generate_title_from_content(self, content: str, max_length: int = 50) -> str:
        """
        从内容生成标题
        
        策略：
        1. 优先从 user 的第一条消息取（通常是查询主题）
        2. 如果没有好的 user 消息，再从 assistant 取第一条有意义的回复
        3. 过滤掉太短的问候语
        
        Args:
            content: 内容文本
            max_length: 标题最大长度
            
        Returns:
            生成的标题
        """
        lines = content.strip().split('\n')
        
        # 先找 user 的第一条消息
        for line in lines:
            line = line.strip()
            if line.startswith('[用户]'):
                # 去掉 [用户] 前缀
                text = line.split(']', 1)[1].strip() if ']' in line else line
                # 过滤太短的问候
                if len(text) >= 3:
                    return text[:max_length]
        
        # 再找 assistant 的第一条有意义的回复
        for line in lines:
            line = line.strip()
            if line.startswith('[AI]'):
                text = line.split(']', 1)[1].strip() if ']' in line else line
                # 过滤掉 "让我尝试..." 这种过渡语句
                skip_prefixes = ['让我', '我来', '正在', '正在尝试', '稍等', '好的']
                if any(text.startswith(p) for p in skip_prefixes):
                    continue
                if len(text) >= 5:
                    return text[:max_length]
        
        # 兜底：取第一条有意义的行
        for line in lines:
            line = line.strip()
            if ']' in line:
                line = line.split(']', 1)[1].strip()
            if len(line) >= 3:
                return line[:max_length]
        
        # 默认标题
        now = datetime.now()
        return f"对话记录 {now.strftime('%m-%d %H:%M')}"
    
    def add_memory_from_messages(
        self,
        messages: List[Dict[str, Any]],
        title: Optional[str] = None,
        source: str = "auto-save",
        base_tags: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[MemoryNode]:
        """
        从对话消息添加记忆（自动检测 skills、格式化内容、生成标签）
        
        这是添加对话记忆的便捷方法，会自动：
        1. 检测 messages 中使用的 skills
        2. 格式化为 [用户]/[AI] 标准格式
        3. 如果没有提供 title，自动生成
        4. 使用 TagGenerator 从内容生成相关标签
        5. 将 skills 作为标签添加
        6. 保存到记忆系统
        
        Args:
            messages: 对话消息列表，每个消息是 dict，包含 role 和 content
            title: 可选，自定义标题
            source: 来源标记（auto-save, cli-import 等）
            base_tags: 基础标签列表
            **kwargs: 传递给 add_memory 的其他参数
            
        Returns:
            创建的记忆节点，如果内容为空则返回 None
            
        Example:
            messages = [
                {"role": "user", "content": "帮我搜索新闻"},
                {"role": "assistant", "content": "正在搜索..."},
                {"role": "tool", "content": "name: web-search..."},
            ]
            node = ms.add_memory_from_messages(messages, source="auto-save")
        """
        if not messages:
            return None
        
        # 检测 skills
        detected_skills = self.detect_skills_in_messages(messages)
        
        # 格式化内容
        content = self.format_conversation(messages, detected_skills)
        
        if not content:
            return None
        
        # 生成或检查标题
        if not title:
            title = self.generate_title_from_content(content)
        
        # 构建标签（只包含 base_tags 和自动提取的关键词，不包含 skill）
        # 过滤掉 auto-saved（避免与 source 重复）
        filtered_base_tags = [t for t in (base_tags or []) if t != 'auto-saved']
        tags = list(filtered_base_tags)
        
        # 从内容生成标签（使用 jieba 提取关键词）
        try:
            generated_tags = self._extract_tags_from_text(content, max_tags=8)
            for tag in generated_tags:
                if tag not in tags:
                    tags.append(tag)
        except Exception as e:
            # 标签生成失败不影响主流程
            pass
        
        # 注意：不把 skills 添加到 tags，只追加到内容中的 [使用的技能]
        
        # 从消息中提取最早的时间戳（用于历史数据导入）
        created_at = None
        for msg in messages:
            ts = msg.get('timestamp')
            if ts:
                try:
                    from datetime import datetime
                    msg_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    if created_at is None or msg_time < created_at:
                        created_at = msg_time
                except (ValueError, TypeError):
                    pass
        
        # 保存记忆（传入原始时间戳）
        return self.add_memory(
            content=content,
            title=title,
            tags=tags,
            source=source,
            created_at=created_at,
            **kwargs
        )
    
    def add_memory(
        self,
        content: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        session_id: Optional[str] = None,
        links: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        auto_link: bool = True,
        link_threshold: float = 0.85,
        created_at: Optional[datetime] = None
    ) -> MemoryNode:
        """
        添加一条记忆
        
        Args:
            content: 记忆内容 (Markdown格式)
            title: 标题
            tags: 标签列表
            source: 来源 (cli/telegram/etc)
            session_id: 会话ID
            links: 手动指定的外链URLs
            metadata: 扩展元数据
            auto_link: 是否自动链接到语义相似的现有节点
            link_threshold: 自动链接的相似度阈值
            created_at: 可选，指定创建时间（用于历史数据导入）
            
        Returns:
            创建的记忆节点
        """
        now = datetime.now()
        # 使用指定的时间或当前时间
        node_time = created_at if created_at else now
        
        # 创建节点
        node = MemoryNode(
            id="",  # 稍后生成
            url="",
            created=node_time,
            modified=now,
            content=content,
            title=title,
            tags=tags or [],
            source=source,
            session_id=session_id,
            links=links or [],
            metadata=metadata or {}
        )
        
        # 生成ID和URL
        node.id = node.generate_id()
        node.url = node.generate_url()
        
        # 计算向量嵌入
        embedding = self.embedding_mgr.encode_single(content)
        embedding_file = self.embedding_mgr.save_embedding(node.id, embedding)
        node.embedding_file = os.path.basename(embedding_file)
        
        # 自动链接到语义相似的现有节点（修复：更新图片时也建立链接）
        if auto_link:
            auto_links = self._find_similar_nodes_for_linking(
                node, threshold=link_threshold, max_links=5
            )
            for url in auto_links:
                if url not in node.links:
                    node.links.append(url)
        
        # 保存到存储
        self.storage.save(node)
        
        # 更新被链接节点的 backlinks
        self._update_backlinks(node)
        
        print(f"Added memory: {node.url}")
        print(f"  ID: {node.id}")
        print(f"  Tags: {node.tags}")
        if node.links:
            print(f"  Links: {len(node.links)}")
        
        return node
    
    def _find_similar_nodes_for_linking(
        self, 
        new_node: MemoryNode, 
        threshold: float = 0.85,
        max_links: int = 5
    ) -> List[str]:
        """为新节点找到语义相似的现有节点用于建立链接"""
        all_nodes = self.storage.get_all()
        if not all_nodes:
            return []
        
        new_emb = self.embedding_mgr.encode_single(new_node.content)
        similarities = []
        
        for node in all_nodes:
            if node.id == new_node.id:
                continue
            
            node_emb = self.embedding_mgr.load_embedding(node.id)
            if node_emb is None:
                continue
            
            sim = self.embedding_mgr.compute_similarity(new_emb, node_emb)
            if sim >= threshold:
                similarities.append((node.url, sim))
        
        # 按相似度排序，取前 N 个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [url for url, _ in similarities[:max_links]]
    
    def _update_backlinks(self, node: MemoryNode):
        """更新被链接节点的 backlinks"""
        for target_url in node.links:
            target_node = self.storage.load_by_url(target_url)
            if target_node:
                if node.url not in target_node.backlinks:
                    target_node.backlinks.append(node.url)
                    self.storage.save(target_node)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_tags: Optional[List[str]] = None,
        time_range_days: Optional[int] = None
    ) -> List[SearchResult]:
        """
        搜索记忆
        
        Returns:
            排序后的搜索结果列表
        """
        return self.retrieval.search(
            query=query,
            top_k=top_k,
            filter_tags=filter_tags,
            time_range_days=time_range_days
        )
    
    def search_with_expansion(
        self,
        query: str,
        top_k: int = 10,
        expansion_depth: int = 1,
        filter_tags: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        子图扩散搜索 (Phase 2)
        
        流程：
        1. 语义检索找到种子节点
        2. 通过 links/backlinks 扩展到邻居节点
        3. 在扩展后的候选集中重新排序
        
        效果：召回语义不相似但概念相关的记忆
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            expansion_depth: 链接扩散深度（1-2推荐）
            filter_tags: 按标签过滤
            
        Returns:
            排序后的搜索结果
        """
        # 先过滤候选集（如果有过滤条件）
        if filter_tags:
            candidates = self.storage.get_all()
            candidates = [
                n for n in candidates
                if any(t in n.tags for t in filter_tags)
            ]
            # 临时替换存储为过滤后的（hacky but works）
            original_get_all = self.retrieval.storage.get_all
            self.retrieval.storage.get_all = lambda: candidates
        
        try:
            results = self.retrieval.search_with_expansion(
                query=query,
                top_k=top_k,
                expansion_depth=expansion_depth
            )
        finally:
            if filter_tags:
                self.retrieval.storage.get_all = original_get_all
        
        return results
    
    def get(self, url_or_id: str) -> Optional[MemoryNode]:
        """
        通过URL或ID获取记忆
        """
        # 尝试作为URL
        if url_or_id.startswith("/memory/"):
            return self.storage.load_by_url(url_or_id)
        
        # 尝试作为ID
        return self.storage.load_by_id(url_or_id)
    
    def get_related(self, url: str, top_k: int = 5) -> List[MemoryNode]:
        """
        获取相关记忆
        """
        return self.retrieval.get_related_memories(url, top_k)
    
    def link_memories(self, from_url: str, to_url: str, bidirectional: bool = True):
        """
        手动建立两个记忆之间的链接
        """
        from_node = self.storage.load_by_url(from_url)
        to_node = self.storage.load_by_url(to_url)
        
        if not from_node or not to_node:
            raise ValueError("Node not found")
        
        # 添加链接
        if to_url not in from_node.links:
            from_node.links.append(to_url)
            self.storage.save(from_node)
        
        if bidirectional:
            if from_url not in to_node.links:
                to_node.links.append(from_url)
                self.storage.save(to_node)
        
        print(f"Linked: {from_url} <-> {to_url}")
    
    def build_graph(self, auto_link: bool = True):
        """
        构建记忆图谱并计算PageRank
        
        Args:
            auto_link: 是否自动建立链接（基于相似度和时间）
        """
        nodes = self.storage.get_all()
        
        if auto_link:
            print(f"Building graph with auto-link for {len(nodes)} nodes...")
            self.graph.auto_build_links(nodes)
        else:
            print(f"Building graph for {len(nodes)} nodes...")
            self.graph.build_from_nodes(nodes)
        
        # 计算PageRank
        scores = self.graph.update_pagerank_scores()
        
        # 保存更新
        for node in nodes:
            if node.url in scores:
                node.pagerank = scores[node.url]
                self.storage.save(node)
        
        print(f"PageRank calculated for {len(scores)} nodes")
        
        # 显示Top 5
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 5 by PageRank:")
        for url, score in sorted_scores[:5]:
            print(f"  {url}: {score:.4f}")
        
        return scores
    
    def list_all(self, limit: int = 100) -> List[MemoryNode]:
        """
        列出所有记忆
        """
        nodes = self.storage.get_all()
        return sorted(nodes, key=lambda n: n.created, reverse=True)[:limit]
    
    def delete(self, url_or_id: str) -> bool:
        """
        删除记忆
        """
        node = self.get(url_or_id)
        if node:
            # 删除嵌入文件
            if node.embedding_file:
                emb_path = os.path.join(self.embedding_dir, node.embedding_file)
                if os.path.exists(emb_path):
                    os.remove(emb_path)
            
            # 删除节点
            return self.storage.delete(node.id)
        
        return False
    
    def stats(self) -> Dict:
        """
        获取系统统计信息
        """
        nodes = self.storage.get_all()
        
        return {
            "total_nodes": len(nodes),
            "total_tags": len(set(t for n in nodes for t in n.tags)),
            "avg_pagerank": sum(n.pagerank for n in nodes) / max(len(nodes), 1),
            "memory_dir": self.memory_dir,
            "embedding_dir": self.embedding_dir,
        }


# 便捷函数（向后兼容）
def create_memory_system() -> Memora:
    """创建默认记忆系统实例"""
    return Memora()

# 新名称
MemorySystem = Memora  # 向后兼容别名


if __name__ == "__main__":
    # 简单测试
    ms = Memora()
    
    # 添加测试记忆
    node1 = ms.add_memory(
        content="VRM动捕项目使用MediaPipe Holistic进行姿态估计...",
        title="VRM Motion Capture",
        tags=["project", "vrm", "motion-capture"]
    )
    
    node2 = ms.add_memory(
        content="灵痕小说第7章需要修正世界观设定...",
        title="灵痕写作",
        tags=["project", "novel", "灵痕"]
    )
    
    # 构建图谱
    ms.build_graph(auto_link=True)
    
    # 搜索
    results = ms.search("VRM", top_k=5)
    print("\nSearch results for 'VRM':")
    for r in results:
        print(f"  {r.node.title}: {r.final_score:.3f}")
    
    # 统计
    print("\nStats:", ms.stats())
