"""
Retrieval Engine - Hybrid Search
检索引擎 - 混合搜索
结合语义相似度、PageRank和时效性

优化版本：两阶段检索
- Stage 1: TF-IDF + jieba 快速召回 top-K 候选
- Stage 2: sentence-transformers 语义精排

时效性策略（重要修复）：
- 不再将"新鲜度"作为正向加分
- 而是作为"遗忘惩罚"仅应用于低重要性内容
- 高PageRank内容不受时间影响（核心知识永不过期）
"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

from .models import MemoryNode, SearchResult
from .storage import MemoryStorage
from .embeddings import (
    get_embedding_manager, EmbeddingManager, TFIDFEmbeddingManager
)
from .pagerank import MemoryGraph
from .config import (
    WEIGHT_SEMANTIC, WEIGHT_PAGERANK, WEIGHT_RECENCY,
    RECENCY_HALF_LIFE, SIMILARITY_THRESHOLD_LOW,
    PAGERANK_IMPORTANCE_THRESHOLD
)


class MemoryRetrieval:
    """
    记忆检索引擎
    
    检索公式（修复后）：
    score = 0.7 × semantic_similarity + 0.3 × pagerank
    
    时效性只用于：
    1. 过滤极端过期的低重要性内容（可选）
    2. 完全不参与排序加分
    """
    
    def __init__(
        self,
        storage: MemoryStorage,
        embedding_mgr: Optional[EmbeddingManager] = None
    ):
        self.storage = storage
        self.embedding_mgr = embedding_mgr or get_embedding_manager()
        self.graph = MemoryGraph()
        self._cache: dict = {}  # 简单缓存
    
    def _calculate_recency_penalty(self, node: MemoryNode, query: str = "") -> float:
        """
        计算时效性惩罚
        
        Returns:
            惩罚因子（1.0 = 无惩罚）
        """
        days_old = (datetime.now() - node.created).days
        
        if days_old < 7:
            base_penalty = 0.8
        elif days_old < 30:
            base_penalty = 0.85
        elif days_old < 365:
            base_penalty = 1.00
        else:
            base_penalty = 0.95
        
        return base_penalty
    
    def _normalize_pagerank(self, pagerank: float, all_pageranks: List[float]) -> float:
        """
        归一化PageRank分数到[0,1]
        
        使用log变换压缩差距，避免新节点(PR=0.5)完全压制旧节点(PR~0.002)
        """
        import math
        if not all_pageranks:
            return 1.0
        max_pr = max(all_pageranks)
        if max_pr <= 0:
            return 1.0
        
        # Log变换：压缩极端差距
        # PR=0.5 -> log(1.5)=0.405, PR=0.002 -> log(1.002)=0.002
        # 归一化后：0.405/0.405=1.0, 0.002/0.405=0.005 -> 差距200倍压缩到200倍？
        # 等等，让我重新算...
        # 实际上 log(1+x) 对小x近似线性，对大x压缩
        # PR=0.5: log(1.5)=0.405
        # PR=0.002: log(1.002)=0.002
        # 还是差距大...
        
        # 改用 soft scaling：sqrt 或 pow(0.5)
        # 平方根变换：sqrt(0.5)=0.707, sqrt(0.002)=0.045 -> 差距14倍（比200倍好）
        return math.sqrt(pagerank) / math.sqrt(max_pr)
    
    def _keyword_match_score(self, query: str, node: MemoryNode) -> float:
        """
        计算关键词匹配分数
        如果查询词在标题/内容中出现，给予额外加权
        """
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        title = (node.title or "").lower()
        content = node.content.lower()
        tags = [str(t).lower() for t in node.tags if t is not None]
        
        score = 0.0
        
        # 标题匹配（最高权重）
        if query_lower in title:
            score += 0.3
        
        # 内容匹配
        if query_lower in content:
            score += 0.2
        
        # 标签匹配
        if any(query_lower in t for t in tags):
            score += 0.25
        
        # 词项匹配（部分匹配）
        for term in query_terms:
            if len(term) >= 2:  # 只考虑长度>=2的词
                if term in title:
                    score += 0.1
                if term in content:
                    score += 0.05
        
        return min(score, 1.0)  # 上限1.0
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_tags: Optional[List[str]] = None,
        time_range_days: Optional[int] = None
    ) -> List[SearchResult]:
        """
        搜索记忆
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            filter_tags: 按标签过滤
            time_range_days: 只搜索最近N天的记忆（默认不过滤）
            
        Returns:
            排序后的搜索结果
        """
        # 1. 获取候选节点
        candidates = self.storage.get_all()
        
        # 2. 过滤
        if filter_tags:
            candidates = [
                n for n in candidates 
                if any(str(t) in filter_tags for t in n.tags if t is not None)
            ]
        
        if time_range_days:
            cutoff = datetime.now() - timedelta(days=time_range_days)
            candidates = [n for n in candidates if n.created >= cutoff]
        
        if not candidates:
            return []
        
        # 3. 计算语义相似度
        query_emb = self.embedding_mgr.encode_single(query)
        
        candidates_with_scores = []
        all_pageranks = [n.pagerank for n in candidates]
        
        for node in candidates:
            # 加载向量
            node_emb = None
            if node.embedding_file:
                node_emb = self.embedding_mgr.load_embedding(node.id)
            
            if node_emb is None:
                # 如果没有预计算的向量，实时编码
                node_emb = self.embedding_mgr.encode_single(node.content)
                # 缓存
                self.embedding_mgr.save_embedding(node.id, node_emb)
                node.embedding_file = f"{node.id}.npy"
            
            # 语义相似度（传递标题+内容进行关键词增强）
            doc_text = f"{node.title or ''} {node.content or ''}"
            semantic_score = self.embedding_mgr.compute_similarity(
                query_emb, node_emb, 
                query_text=query, doc_text=doc_text
            )
            
            # 关键词匹配加分
            keyword_bonus = self._keyword_match_score(query, node)
            # 将关键词匹配融入语义分数（确保精确匹配的内容不会太差）
            semantic_score = max(semantic_score, keyword_bonus * 0.8)
            
            # 过滤低相似度
            if semantic_score < SIMILARITY_THRESHOLD_LOW:
                continue
            
            # PageRank分数（归一化）
            pagerank_score = self._normalize_pagerank(node.pagerank, all_pageranks)
            
            # 时效性惩罚（统一应用于所有节点，不区分重要/不重要）
            recency_penalty = self._calculate_recency_penalty(node, query)
            
            # 综合分数：语义 + PageRank + 时效性惩罚
            # 修复：所有节点统一应用惩罚，新节点(<7天)会被降权
            final_score = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_PAGERANK * pagerank_score
            ) * recency_penalty
            
            candidates_with_scores.append(SearchResult(
                node=node,
                semantic_score=semantic_score,
                pagerank_score=pagerank_score,
                recency_score=recency_penalty,  # 改名：penalty 而非 score
                final_score=final_score
            ))
        
        # 4. 排序
        candidates_with_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        return candidates_with_scores[:top_k]
    
    def search_by_url(self, url: str) -> Optional[MemoryNode]:
        """通过URL精确查找"""
        return self.storage.load_by_url(url)
    
    def search_by_id(self, node_id: str) -> Optional[MemoryNode]:
        """通过ID精确查找"""
        return self.storage.load_by_id(node_id)
    
    def get_related_memories(self, url: str, top_k: int = 5) -> List[MemoryNode]:
        """
        获取相关记忆（通过链接关系）
        """
        node = self.storage.load_by_url(url)
        if not node:
            return []
        
        related = []
        
        # 直接链接的节点
        for link_url in node.links:
            linked_node = self.storage.load_by_url(link_url)
            if linked_node:
                related.append(linked_node)
        
        # 反链节点
        for backlink_url in node.backlinks:
            backlinked_node = self.storage.load_by_url(backlink_url)
            if backlinked_node and backlinked_node not in related:
                related.append(backlinked_node)
        
        # 如果不够，用PageRank补充
        if len(related) < top_k:
            all_nodes = self.storage.get_all()
            # 按PageRank排序，排除已有的
            sorted_nodes = sorted(all_nodes, key=lambda n: n.pagerank, reverse=True)
            for n in sorted_nodes:
                if n not in related and n.url != url:
                    related.append(n)
                if len(related) >= top_k:
                    break
        
        return related[:top_k]
    
    def search_with_expansion(self, query: str, top_k: int = 10, expansion_depth: int = 1) -> List[SearchResult]:
        """
        子图扩散搜索 (Phase 2 实现)
        
        策略：
        1. 先用语义检索找到种子节点 (top_k)
        2. 通过 links/backlinks 扩展到邻居节点
        3. 在扩展后的候选集中重新排序
        
        效果：召回语义不相似但概念相关的记忆
        """
        # 阶段1: 语义检索获取种子
        seed_results = self.search(query, top_k=top_k)
        if not seed_results:
            return []
        
        seed_urls = {r.node.url for r in seed_results}
        candidate_nodes = {r.node: r for r in seed_results}  # node -> original_result
        
        # 阶段2: 链接扩散
        for depth in range(expansion_depth):
            current_urls = set(candidate_nodes.keys())
            new_urls = set()
            
            for url in current_urls:
                node = self.storage.load_by_url(url)
                if not node:
                    continue
                
                # 添加出链（我引用的）
                for link_url in node.links:
                    if link_url not in seed_urls and link_url not in new_urls:
                        linked_node = self.storage.load_by_url(link_url)
                        if linked_node:
                            candidate_nodes[linked_node] = None  # 待计算分数
                            new_urls.add(link_url)
                
                # 添加入链（引用我的）
                for backlink_url in node.backlinks:
                    if backlink_url not in seed_urls and backlink_url not in new_urls:
                        backlinked_node = self.storage.load_by_url(backlink_url)
                        if backlinked_node:
                            candidate_nodes[backlinked_node] = None
                            new_urls.add(backlink_url)
            
            print(f"  [Expansion depth {depth+1}] 新增 {len(new_urls)} 个节点")
            if not new_urls:
                break
        
        # 阶段3: 重新计算所有候选的分数
        query_emb = self.embedding_mgr.encode_single(query)
        all_pageranks = [n.pagerank for n in candidate_nodes.keys()]
        
        results = []
        for node, original_result in candidate_nodes.items():
            if original_result is not None:
                # 种子节点保留原分数
                results.append(original_result)
                continue
            
            # 扩展节点需要计算分数
            node_emb = self.embedding_mgr.load_embedding(node.id)
            if node_emb is None:
                node_emb = self.embedding_mgr.encode_single(node.content)
            
            doc_text = f"{node.title or ''} {node.content or ''}"
            semantic_score = self.embedding_mgr.compute_similarity(
                query_emb, node_emb, query_text=query, doc_text=doc_text
            )
            
            # PageRank
            max_pr = max(all_pageranks) if all_pageranks else 1.0
            pagerank_score = node.pagerank / max_pr if max_pr > 0 else 1.0
            
            days_old = (datetime.now() - node.created).days
            if days_old < 7:
                recency_penalty = 0.8
            elif days_old < 30:
                recency_penalty = 0.85
            elif days_old < 365:
                recency_penalty = 1.00
            else:
                recency_penalty = 0.95
            
            final_score = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_PAGERANK * pagerank_score
            ) * recency_penalty
            
            results.append(SearchResult(
                node=node,
                semantic_score=semantic_score,
                pagerank_score=pagerank_score,
                recency_score=recency_penalty,
                final_score=final_score
            ))
        
        # 排序并返回
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]
    
    def refresh_graph(self):
        """刷新图谱（重新计算PageRank）"""
        nodes = self.storage.get_all()
        self.graph.build_from_nodes(nodes)
        scores = self.graph.update_pagerank_scores()
        
        # 保存更新后的分数
        for node in nodes:
            self.storage.save(node)
        
        return scores


class TwoStageRetriever:
    """
    两阶段检索器
    
    Stage 1: TF-IDF + jieba 快速召回
    - 使用轻量级的 TF-IDF 向量快速筛选候选
    - 召回 top-K (默认 100) 个候选
    
    Stage 2: sentence-transformers 语义精排
    - 对 Stage 1 的候选进行深度语义编码
    - 结合 PageRank 计算最终分数（时效性仅作为遗忘惩罚）
    
    关键修正：
    - 久远的重要数据应该更容易被回忆
    - 时效性只惩罚低重要性内容
    """
    
    def __init__(
        self,
        storage: MemoryStorage,
        semantic_mgr: Optional[EmbeddingManager] = None,
        first_stage_k: int = 100,
        use_tfidf: bool = True
    ):
        self.storage = storage
        self.semantic_mgr = semantic_mgr or get_embedding_manager('auto')
        self.first_stage_k = first_stage_k
        self.use_tfidf = use_tfidf
        
        # 初始化 TF-IDF 管理器（用于第一阶段）
        self.tfidf_mgr = TFIDFEmbeddingManager(max_features=5000)
        self._tfidf_fitted = False
        self._node_id_to_idx: Dict[str, int] = {}
        self._node_embeddings: Dict[str, np.ndarray] = {}
    
    def _build_tfidf_index(self, nodes: List[MemoryNode]):
        """构建 TF-IDF 索引（一次性）"""
        if self._tfidf_fitted:
            return
        
        texts = [self._node_to_text(n) for n in nodes]
        self.tfidf_mgr.fit(texts)
        
        # 预计算所有节点的 TF-IDF 向量
        for i, node in enumerate(nodes):
            self._node_id_to_idx[node.id] = i
            vec = self.tfidf_mgr.encode_single(self._node_to_text(node))
            self._node_embeddings[node.id] = vec
        
        self._tfidf_fitted = True
        print(f"TF-IDF index built for {len(nodes)} nodes")
    
    def _node_to_text(self, node: MemoryNode) -> str:
        """将节点转换为文本（用于向量化）"""
        parts = []
        if node.title:
            parts.append(node.title)
        parts.append(node.content)
        if node.tags:
            parts.extend(str(t) for t in node.tags if t is not None)
        return ' '.join(parts)
    
    def _first_stage_recall(
        self,
        query: str,
        candidates: List[MemoryNode],
        top_k: int
    ) -> List[Tuple[MemoryNode, float]]:
        """
        第一阶段：TF-IDF 快速召回
        
        Returns:
            [(node, tfidf_score), ...] 按分数排序
        """
        # 确保索引已构建
        self._build_tfidf_index(candidates)
        
        # 编码查询
        query_vec = self.tfidf_mgr.encode_single(query)
        
        # 计算与所有候选的相似度
        scored_candidates = []
        for node in candidates:
            node_vec = self._node_embeddings.get(node.id)
            if node_vec is not None:
                # 余弦相似度
                norm_q = np.linalg.norm(query_vec)
                norm_n = np.linalg.norm(node_vec)
                if norm_q > 0 and norm_n > 0:
                    sim = np.dot(query_vec, node_vec) / (norm_q * norm_n)
                    # 归一化到 [0, 1]
                    sim = (sim + 1) / 2
                else:
                    sim = 0.0
            else:
                sim = 0.0
            
            # 关键词匹配增强
            keyword_score = self._fast_keyword_score(query, node)
            final_sim = max(sim, keyword_score * 0.5)
            
            scored_candidates.append((node, final_sim))
        
        # 排序并返回 top-k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]
    
    def _fast_keyword_score(self, query: str, node: MemoryNode) -> float:
        """快速关键词匹配分数（用于第一阶段）"""
        query_lower = query.lower().strip()
        score = 0.0
        
        # 标题匹配（最高权重）
        if node.title and query_lower in node.title.lower():
            score += 1.0
        
        # 内容精确匹配 - 如果查询文本在内容中出现，给予高分
        content_lower = node.content.lower()
        if query_lower in content_lower:
            # 计算匹配位置的内容密度（前面部分匹配权重更高）
            match_pos = content_lower.find(query_lower)
            position_weight = max(0.3, 1.0 - (match_pos / len(content_lower)) * 0.5)
            score += 0.8 * position_weight
        
        # 检查关键术语匹配（查询中的每个重要词）
        query_terms = [t for t in query_lower.split() if len(t) >= 4]  # 只考虑长度>=4的词
        if query_terms:
            matched_terms = sum(1 for term in query_terms if term in content_lower)
            term_coverage = matched_terms / len(query_terms)
            score += 0.3 * term_coverage
        
        # 标签匹配
        if any(query_lower in str(t).lower() for t in node.tags if t is not None):
            score += 0.8
        
        return min(score, 1.5)  # 上限1.5，允许超额匹配
    
    def _is_exact_keyword_match(self, query: str, node: MemoryNode) -> bool:
        """
        检测是否关键词精确匹配
        - 查询词完整出现在标题中，或
        - 查询词完整出现在内容开头200字内
        """
        query_lower = query.lower().strip()
        
        # 标题精确匹配
        if node.title and query_lower in node.title.lower():
            return True
        
        # 内容开头精确匹配（前200字）
        content_preview = node.content[:200].lower()
        if query_lower in content_preview:
            return True
        
        return False
    
    def _second_stage_rank(
        self,
        query: str,
        first_stage_results: List[Tuple[MemoryNode, float]],
        all_pageranks: List[float]
    ) -> List[SearchResult]:
        """
        第二阶段：语义精排（修复时效性逻辑）
        
        关键修正：
        - 时效性不再是正向加分
        - 而是遗忘惩罚，仅应用于低重要性内容
        - 高PageRank内容（核心知识）不受时间影响
        """
        if not first_stage_results:
            return []
        
        # 编码查询（一次）
        query_emb = self.semantic_mgr.encode_single(query)
        
        results = []
        for node, tfidf_score in first_stage_results:
            # 获取或计算节点的语义向量
            node_emb = None
            if node.embedding_file:
                node_emb = self.semantic_mgr.load_embedding(node.id)
            
            if node_emb is None:
                # 实时编码
                node_emb = self.semantic_mgr.encode_single(node.content)
                self.semantic_mgr.save_embedding(node.id, node_emb)
                node.embedding_file = f"{node.id}.npy"
            
            # 语义相似度（使用标题+内容）
            doc_text = f"{node.title or ''} {node.content or ''}"
            semantic_score = self.semantic_mgr.compute_similarity(
                query_emb, node_emb,
                query_text=query, doc_text=doc_text
            )
            
            # === 规则1: 关键词精确匹配时，强制提升语义分数到 0.7+ ===
            keyword_score = self._fast_keyword_score(query, node)
            exact_match = self._is_exact_keyword_match(query, node)
            
            if exact_match or keyword_score >= 0.8:
                # 关键词精确匹配或高匹配，强制语义分数 ≥0.7
                semantic_score = max(semantic_score, 0.7)
            elif keyword_score >= 0.5:
                # 部分匹配，确保语义分数 ≥0.5
                semantic_score = max(semantic_score, 0.5)
            
            # PageRank 分数（归一化：保留相对比例，最小值不压到0）
            max_pr = max(all_pageranks) if all_pageranks else 1.0
            if max_pr > 0:
                pagerank_score = node.pagerank / max_pr
            else:
                pagerank_score = 1.0
            
            days_old = (datetime.now() - node.created).days
            if days_old < 7:
                base_penalty = 0.8 if keyword_score < 0.8 else 0.85
            elif days_old < 30:
                base_penalty = 0.85 if keyword_score < 0.8 else 0.95
            elif days_old < 365:
                base_penalty = 1.00
            else:
                base_penalty = 0.95
            
            recency_penalty = base_penalty
            
            # 计算最终分数
            base_score = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_PAGERANK * pagerank_score +
                0.1 * tfidf_score
            )
            final_score = base_score * recency_penalty
            
            results.append(SearchResult(
                node=node,
                semantic_score=semantic_score,
                pagerank_score=pagerank_score,
                recency_score=recency_penalty,  # 改名为惩罚
                final_score=min(final_score, 1.0)
            ))
        
        # 按最终分数排序
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_tags: Optional[List[str]] = None,
        time_range_days: Optional[int] = None
    ) -> List[SearchResult]:
        """
        混合检索（Hybrid）- 默认检索方法
        
        流程：TF-IDF召回 → 子图扩散 → 语义精排
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            filter_tags: 按标签过滤
            time_range_days: 只搜索最近N天的记忆（默认不过滤）
            
        Returns:
            排序后的搜索结果
        """
        # 默认使用 hybrid 方法（带图扩散）
        return self.search_with_graph_expansion(
            query=query,
            top_k=top_k,
            filter_tags=filter_tags,
            time_range_days=time_range_days
        )
    
    def search_basic(
        self,
        query: str,
        top_k: int = 10,
        filter_tags: Optional[List[str]] = None,
        time_range_days: Optional[int] = None
    ) -> List[SearchResult]:
        """
        基础两阶段检索（无图扩散）- 用于对比测试
        
        流程：TF-IDF召回 → 语义精排
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            filter_tags: 按标签过滤
            time_range_days: 只搜索最近N天的记忆（默认不过滤）
            
        Returns:
            排序后的搜索结果
        """
        # 1. 获取候选节点
        candidates = self.storage.get_all()
        
        # 2. 过滤
        if filter_tags:
            candidates = [
                n for n in candidates
                if any(str(t) in filter_tags for t in n.tags if t is not None)
            ]
        
        if time_range_days:
            cutoff = datetime.now() - timedelta(days=time_range_days)
            candidates = [n for n in candidates if n.created >= cutoff]
        
        if not candidates:
            return []
        
        # 3. 第一阶段：TF-IDF 快速召回
        all_pageranks = [n.pagerank for n in candidates]
        
        if self.use_tfidf and len(candidates) > self.first_stage_k:
            first_stage_results = self._first_stage_recall(
                query, candidates, self.first_stage_k
            )
        else:
            # 候选数量少，跳过第一阶段
            first_stage_results = [(n, 0.5) for n in candidates]
        
        # 4. 第二阶段：语义精排
        results = self._second_stage_rank(
            query, first_stage_results, all_pageranks
        )
        
        return results[:top_k]
    
    def refresh_index(self):
        """刷新 TF-IDF 索引（当有新节点时调用）"""
        self._tfidf_fitted = False
        self._node_id_to_idx.clear()
        self._node_embeddings.clear()
        print("TF-IDF index cleared, will rebuild on next search")
    
    def search_with_expansion(
        self, 
        query: str, 
        top_k: int = 10, 
        seed_k: int = 5,
        expansion_depth: int = 1,
        max_expanded: int = 50,
        expansion_boost: float = 0.9  # 扩散节点的分数折扣
    ) -> List[SearchResult]:
        """
        两阶段检索 + 子图扩散 (Phase 2 实现 - 修复版)
        
        核心策略：
        1. 两阶段检索获取高质量种子节点 (seed_k个)
        2. 通过 links/backlinks 扩展到邻居节点
        3. 种子节点保留原分数，扩展节点打折
        4. 在扩展集合中重排序
        
        Args:
            seed_k: 种子节点数（语义检索top-k）
            expansion_depth: 扩散深度（建议1-2）
            max_expanded: 最大扩展节点数
            expansion_boost: 扩散节点分数乘数（<1表示降权）
        
        Returns:
            排序后的搜索结果，包含种子节点和扩散节点
        """
        from datetime import datetime
        
        # 阶段1: 两阶段检索获取种子
        print(f"[子图扩散] 阶段1: 语义检索 top-{seed_k} 种子节点...")
        seed_results = self.search(query, top_k=seed_k)
        if not seed_results:
            return []
        
        seed_nodes = {r.node.url: r for r in seed_results}
        print(f"  找到 {len(seed_nodes)} 个种子节点")
        
        # 阶段2: 链接扩散
        print(f"[子图扩散] 阶段2: 扩散深度={expansion_depth}, 最大扩展={max_expanded}...")
        expanded_nodes = {}  # url -> node
        expanded_from = {}   # url -> list of seed_urls (用于调试)
        
        for depth in range(expansion_depth):
            current_seeds = list(seed_nodes.keys()) if depth == 0 else list(expanded_nodes.keys())
            new_count = 0
            
            for seed_url in current_seeds:
                node = seed_nodes[seed_url].node if seed_url in seed_nodes else expanded_nodes[seed_url]
                
                # 收集所有链接（出链+入链）
                all_links = list(node.links) + list(node.backlinks)
                
                for link_url in all_links:
                    # 跳过已在种子中的
                    if link_url in seed_nodes:
                        continue
                    # 跳过已扩展的
                    if link_url in expanded_nodes:
                        continue
                    # 限制扩展数量
                    if len(expanded_nodes) >= max_expanded:
                        break
                    
                    # 加载邻居节点
                    linked_node = self.storage.load_by_url(link_url)
                    if linked_node:
                        expanded_nodes[link_url] = linked_node
                        expanded_from[link_url] = seed_url
                        new_count += 1
                
                if len(expanded_nodes) >= max_expanded:
                    break
            
            print(f"  [深度 {depth+1}] 新增 {new_count} 个节点，累计扩展 {len(expanded_nodes)}")
            if new_count == 0:
                break
        
        # 阶段3: 对扩展节点计算分数
        print(f"[子图扩散] 阶段3: 计算 {len(expanded_nodes)} 个扩展节点的分数...")
        
        all_candidates = list(seed_nodes.values())  # SearchResult列表
        all_pageranks = [r.node.pagerank for r in all_candidates] + [n.pagerank for n in expanded_nodes.values()]
        
        # 编码查询（只编码一次）
        query_emb = self.semantic_mgr.encode_single(query)
        
        for url, node in expanded_nodes.items():
            # 加载/计算向量
            node_emb = self.semantic_mgr.load_embedding(node.id)
            if node_emb is None:
                node_emb = self.semantic_mgr.encode_single(node.content)
                self.semantic_mgr.save_embedding(node.id, node_emb)
            
            # 语义相似度
            doc_text = f"{node.title or ''} {node.content or ''}"
            semantic_score = self.semantic_mgr.compute_similarity(
                query_emb, node_emb, query_text=query, doc_text=doc_text
            )
            
            # 关键词匹配（扩展节点也需要）
            keyword_score = self._fast_keyword_score(query, node)
            exact_match = self._is_exact_keyword_match(query, node)
            
            if exact_match or keyword_score >= 0.8:
                semantic_score = max(semantic_score, 0.7)
            elif keyword_score >= 0.5:
                semantic_score = max(semantic_score, 0.5)
            
            # PageRank
            max_pr = max(all_pageranks) if all_pageranks else 1.0
            pagerank_score = node.pagerank / max_pr if max_pr > 0 else 1.0
            
            
            days_old = (datetime.now() - node.created).days
            if days_old < 7:
                base_penalty = 0.8 if keyword_score < 0.8 else 0.85
            elif days_old < 30:
                base_penalty = 0.85 if keyword_score < 0.8 else 0.95
            elif days_old < 365:
                base_penalty = 1.00
            else:
                base_penalty = 0.95
            
            recency_penalty = min(1.0, base_penalty)
            
            # 综合分数（扩展节点应用折扣）
            base_score = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_PAGERANK * pagerank_score
            )
            final_score = base_score * recency_penalty * expansion_boost
            
            all_candidates.append(SearchResult(
                node=node,
                semantic_score=semantic_score,
                pagerank_score=pagerank_score,
                recency_score=recency_penalty,
                final_score=min(final_score, 1.0),
                # 标记为扩展节点（可选，可用于调试）
                metadata={'expanded_from': expanded_from.get(url), 'is_expanded': True}
            ))
        
        # 阶段4: 重排序并返回
        all_candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # 统计信息
        top_results = all_candidates[:top_k]
        seed_in_top = sum(1 for r in top_results if r.node.url in seed_nodes)
        expanded_in_top = len(top_results) - seed_in_top
        
        print(f"[子图扩散] 完成: 返回 top-{top_k} (种子{seed_in_top}, 扩散{expanded_in_top})")
        
        return top_results
    
    def search_with_graph_expansion(
        self,
        query: str,
        top_k: int = 10,
        recall_k: int = 10,          # TF-IDF 召回数量（默认10，给扩散留空间）
        expansion_depth: int = 1,    # 扩散深度
        max_expanded: int = 50,      # 最大扩展节点数
        expansion_boost: float = 0.85, # 扩展节点分数折扣
        filter_tags: Optional[List[str]] = None,
        time_range_days: Optional[int] = None
    ) -> List[SearchResult]:
        """
        混合检索：TF-IDF 召回 + 子图扩散 + 语义精排
        
        核心流程：
        1. TF-IDF 快速召回 top-recall_k 候选（全库扫描）
        2. 在这 recall_k 个候选中，通过 links/backlinks 扩散到邻居
        3. 对扩展后的集合（候选+邻居）进行语义精排
        
        优势：
        - 比纯语义扩散快（TF-IDF 召回比语义编码快10倍+）
        - 比纯 TF-IDF 召回全（扩散能发现关联内容）
        - 适合中等长度查询（5-20字）
        
        Args:
            recall_k: TF-IDF 第一阶段召回数量（默认10，平衡召回与扩散）
            expansion_depth: 图扩散深度（建议1，2开始变慢）
            max_expanded: 最大扩展节点数（防止爆炸）
            expansion_boost: 扩展节点分数折扣（<1降权）
        
        Returns:
            排序后的搜索结果
        """
        from datetime import datetime, timedelta
        
        print(f"[混合检索] 查询: '{query}'")
        print(f"[混合检索] 策略: TF-IDF召回{recall_k} → 扩散 → 语义精排")
        
        # === 阶段1: TF-IDF 快速召回 top-recall_k ===
        print(f"[阶段1/4] TF-IDF 快速召回 top-{recall_k}...")
        
        candidates = self.storage.get_all()
        
        # 过滤
        if filter_tags:
            candidates = [
                n for n in candidates
                if any(str(t) in filter_tags for t in n.tags if t is not None)
            ]
        
        if time_range_days:
            cutoff = datetime.now() - timedelta(days=time_range_days)
            candidates = [n for n in candidates if n.created >= cutoff]
        
        if not candidates:
            return []
        
        # 构建/使用 TF-IDF 索引
        self._build_tfidf_index(candidates)
        query_vec = self.tfidf_mgr.encode_single(query)
        
        # 计算所有节点的 TF-IDF 分数
        scored_candidates = []
        for node in candidates:
            node_vec = self._node_embeddings.get(node.id)
            if node_vec is not None:
                norm_q = np.linalg.norm(query_vec)
                norm_n = np.linalg.norm(node_vec)
                if norm_q > 0 and norm_n > 0:
                    tfidf_sim = np.dot(query_vec, node_vec) / (norm_q * norm_n)
                    tfidf_sim = (tfidf_sim + 1) / 2  # 归一化到[0,1]
                else:
                    tfidf_sim = 0.0
            else:
                tfidf_sim = 0.0
            
            # 关键词匹配增强
            keyword_score = self._fast_keyword_score(query, node)
            final_tfidf = max(tfidf_sim, keyword_score * 0.5)
            
            scored_candidates.append((node, final_tfidf))
        
        # 排序取 top-recall_k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        recall_candidates = scored_candidates[:recall_k]
        
        recall_urls = {n.url for n, _ in recall_candidates}
        print(f"  TF-IDF 召回 {len(recall_candidates)} 个候选")
        
        # === 阶段2: 在召回候选中扩散 ===
        print(f"[阶段2/4] 图扩散: 深度={expansion_depth}, 最大{max_expanded}...")
        
        expanded_nodes = {}  # url -> node
        expanded_from = {}   # url -> source_url
        
        for depth in range(expansion_depth):
            # 当前层：第一层用recall候选，后续用新扩展的
            if depth == 0:
                current_layer = [n for n, _ in recall_candidates]
            else:
                current_layer = list(expanded_nodes.values())
            
            new_count = 0
            for node in current_layer:
                all_links = list(node.links) + list(node.backlinks)
                
                for link_url in all_links:
                    # 跳过已在召回集中的
                    if link_url in recall_urls:
                        continue
                    # 跳过已扩展的
                    if link_url in expanded_nodes:
                        continue
                    # 限制数量
                    if len(expanded_nodes) >= max_expanded:
                        break
                    
                    linked_node = self.storage.load_by_url(link_url)
                    if linked_node:
                        # 时间过滤：扩展节点也要符合时间范围
                        if time_range_days and linked_node.created < cutoff:
                            continue
                        expanded_nodes[link_url] = linked_node
                        expanded_from[link_url] = node.url
                        new_count += 1
                
                if len(expanded_nodes) >= max_expanded:
                    break
            
            print(f"  [深度 {depth+1}] 新增 {new_count} 个节点")
            if new_count == 0:
                break
        
        print(f"  扩散完成: 召回{len(recall_candidates)} + 扩展{len(expanded_nodes)} = {len(recall_candidates) + len(expanded_nodes)} 个节点")
        
        # === 阶段3: 语义精排 ===
        print(f"[阶段3/4] 语义精排...")
        
        # 合并所有候选（召回+扩展）
        all_nodes = [n for n, _ in recall_candidates] + list(expanded_nodes.values())
        all_pageranks = [n.pagerank for n in all_nodes]
        
        # 编码查询（一次）
        query_emb = self.semantic_mgr.encode_single(query)
        
        results = []
        
        # 处理召回候选（保留原分数权重）
        for node, tfidf_score in recall_candidates:
            node_emb = self.semantic_mgr.load_embedding(node.id)
            if node_emb is None:
                node_emb = self.semantic_mgr.encode_single(node.content)
                self.semantic_mgr.save_embedding(node.id, node_emb)
            
            doc_text = f"{node.title or ''} {node.content or ''}"
            semantic_score = self.semantic_mgr.compute_similarity(
                query_emb, node_emb, query_text=query, doc_text=doc_text
            )
            
            # 关键词匹配提升
            keyword_score = self._fast_keyword_score(query, node)
            exact_match = self._is_exact_keyword_match(query, node)
            if exact_match or keyword_score >= 0.8:
                semantic_score = max(semantic_score, 0.7)
            elif keyword_score >= 0.5:
                semantic_score = max(semantic_score, 0.5)
            
            # PageRank
            max_pr = max(all_pageranks) if all_pageranks else 1.0
            pagerank_score = node.pagerank / max_pr if max_pr > 0 else 1.0
            
            
            days_old = (datetime.now() - node.created).days
            if days_old < 7:
                base_penalty = 0.8 if keyword_score < 0.8 else 0.85
            elif days_old < 30:
                base_penalty = 0.85 if keyword_score < 0.8 else 0.95
            elif days_old < 365:
                base_penalty = 1.00
            else:
                base_penalty = 0.95
            
            recency_penalty = min(1.0, base_penalty)
            
            # 综合分数（召回候选保留 TF-IDF 权重）
            base_score = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_PAGERANK * pagerank_score +
                0.1 * tfidf_score  # TF-IDF 分数小幅加权
            )
            final_score = base_score * recency_penalty
            
            results.append(SearchResult(
                node=node,
                semantic_score=semantic_score,
                pagerank_score=pagerank_score,
                recency_score=recency_penalty,
                final_score=min(final_score, 1.0),
                metadata={'source': 'recall', 'tfidf_score': tfidf_score}
            ))
        
        # 处理扩展节点（应用折扣）
        for url, node in expanded_nodes.items():
            node_emb = self.semantic_mgr.load_embedding(node.id)
            if node_emb is None:
                node_emb = self.semantic_mgr.encode_single(node.content)
                self.semantic_mgr.save_embedding(node.id, node_emb)
            
            doc_text = f"{node.title or ''} {node.content or ''}"
            semantic_score = self.semantic_mgr.compute_similarity(
                query_emb, node_emb, query_text=query, doc_text=doc_text
            )
            
            keyword_score = self._fast_keyword_score(query, node)
            exact_match = self._is_exact_keyword_match(query, node)
            if exact_match or keyword_score >= 0.8:
                semantic_score = max(semantic_score, 0.7)
            elif keyword_score >= 0.5:
                semantic_score = max(semantic_score, 0.5)
            
            max_pr = max(all_pageranks) if all_pageranks else 1.0
            pagerank_score = node.pagerank / max_pr if max_pr > 0 else 1.0
            
            
            days_old = (datetime.now() - node.created).days
            if days_old < 7:
                base_penalty = 0.8 if keyword_score < 0.8 else 0.85
            elif days_old < 30:
                base_penalty = 0.85 if keyword_score < 0.8 else 0.95
            elif days_old < 365:
                base_penalty = 1.00
            else:
                base_penalty = 0.95
            
            recency_penalty = min(1.0, base_penalty)
            
            # 扩展节点打折
            base_score = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_PAGERANK * pagerank_score
            )
            final_score = base_score * recency_penalty * expansion_boost
            
            results.append(SearchResult(
                node=node,
                semantic_score=semantic_score,
                pagerank_score=pagerank_score,
                recency_score=recency_penalty,
                final_score=min(final_score, 1.0),
                metadata={
                    'source': 'expanded', 
                    'expanded_from': expanded_from.get(url),
                    'is_expanded': True
                }
            ))
        
        # === 阶段4: 排序返回 ===
        print(f"[阶段4/4] 排序返回 top-{top_k}...")
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        top_results = results[:top_k]
        recall_in_top = sum(1 for r in top_results if r.metadata.get('source') == 'recall')
        expanded_in_top = len(top_results) - recall_in_top
        
        print(f"[混合检索] 完成: 返回 top-{top_k} (召回{recall_in_top}, 扩散{expanded_in_top})")
        
        return top_results


class MemoryIndex:
    """
    记忆索引管理器
    用于快速文本搜索
    """
    
    def __init__(self, storage: MemoryStorage, index_dir: str):
        self.storage = storage
        self.index_dir = index_dir
    
    def full_text_search(self, keywords: str, top_k: int = 10) -> List[MemoryNode]:
        """
        全文搜索（使用ripgrep或简单扫描）
        """
        import subprocess
        import re
        
        nodes = []
        
        # 尝试使用ripgrep
        try:
            result = subprocess.run(
                ["rg", "-i", "-l", keywords, str(self.storage.base_dir)],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                file_paths = result.stdout.strip().split("\n")
                for path in file_paths[:top_k]:
                    if path:
                        node = self.storage.load(path)
                        if node:
                            nodes.append(node)
                return nodes
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fallback: 简单扫描
        pattern = re.compile(keywords, re.IGNORECASE)
        for node in self.storage.iterate_all():
            if pattern.search(node.content) or pattern.search(node.title or ""):
                nodes.append(node)
            if len(nodes) >= top_k:
                break
        
        return nodes
