"""
自适应检索 - Adaptive Retrieval

核心策略：
1. 分析查询特征（长度、关键词密度、语义丰富度）
2. 动态调整召回策略和扩散参数
3. 智能融合多路召回结果

对比混合检索的固定参数，自适应检索能根据查询质量自动优化
"""
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .models import MemoryNode, SearchResult
from .storage import MemoryStorage
from .embeddings import EmbeddingManager, get_embedding_manager, TFIDFEmbeddingManager
from .retrieval import TwoStageRetriever
from .config import WEIGHT_SEMANTIC, WEIGHT_PAGERANK


@dataclass
class QueryFeatures:
    """查询特征分析结果"""
    length: int              # 查询长度
    word_count: int          # 词数（中文按字，英文按词）
    has_numbers: bool        # 是否包含数字
    has_dates: bool          # 是否包含日期
    keyword_density: float   # 关键词密度（实词占比）
    is_fuzzy: bool           # 是否模糊查询（那个/这个/怎么）
    is_specific: bool        # 是否具体查询（包含专有名词）
    
    def __repr__(self):
        return f"QueryFeatures(len={self.length}, words={self.word_count}, " \
               f"fuzzy={self.is_fuzzy}, specific={self.is_specific})"


class AdaptiveRetriever(TwoStageRetriever):
    """
    自适应检索器
    
    继承 TwoStageRetriever，添加自适应策略：
    - 查询特征分析
    - 动态参数调整
    - 多路召回融合
    """
    
    def __init__(self, storage: MemoryStorage, semantic_mgr: Optional[EmbeddingManager] = None):
        super().__init__(storage, semantic_mgr)
        
        # 模糊词列表
        self.fuzzy_words = {'那个', '这个', '怎么', '怎样', '如何', '什么', '哪里', '哪个'}
        # 专有名词指示词
        self.specific_indicators = {'bug', 'fix', 'pr', 'page', 'rank', 'test', 'api', '数据', '项目'}
    
    def _analyze_query(self, query: str) -> QueryFeatures:
        """
        分析查询特征
        
        Returns:
            QueryFeatures: 查询特征对象
        """
        # 基础特征
        length = len(query)
        words = query.split()
        word_count = len(words)
        
        # 是否包含数字
        has_numbers = any(c.isdigit() for c in query)
        
        # 是否包含日期模式
        import re
        date_patterns = [r'\d{4}[-/年]', r'\d{1,2}[-/月]', r'周[一二三四五六日]']
        has_dates = any(re.search(p, query) for p in date_patterns)
        
        # 关键词密度（简单估计：非停用词占比）
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        content_words = [w for w in query if w not in stopwords and len(w) >= 2]
        keyword_density = len(content_words) / max(len(query), 1)
        
        # 是否模糊查询
        is_fuzzy = any(fw in query for fw in self.fuzzy_words)
        
        # 是否具体查询
        is_specific = any(si in query.lower() for si in self.specific_indicators)
        is_specific = is_specific or has_numbers or has_dates
        
        return QueryFeatures(
            length=length,
            word_count=word_count,
            has_numbers=has_numbers,
            has_dates=has_dates,
            keyword_density=keyword_density,
            is_fuzzy=is_fuzzy,
            is_specific=is_specific
        )
    
    def _decide_strategy(self, features: QueryFeatures) -> Dict:
        """
        根据查询特征决定检索策略
        
        Returns:
            策略参数字典
        """
        strategy = {
            'use_tfidf': True,
            'use_semantic_seed': False,
            'recall_k': 10,
            'expansion_depth': 1,
            'max_expanded': 30,
            'expansion_boost': 0.85,
            'description': '默认策略'
        }
        
        # 策略1: 超短查询（<=3字）- 必须扩散
        if features.length <= 3:
            strategy.update({
                'use_tfidf': False,
                'use_semantic_seed': True,
                'recall_k': 5,
                'max_expanded': 50,
                'expansion_boost': 0.9,
                'description': '超短查询-语义种子+强扩散'
            })
        
        # 策略2: 短模糊查询（4-8字，含模糊词）
        elif features.length <= 8 and features.is_fuzzy:
            strategy.update({
                'recall_k': 8,
                'max_expanded': 40,
                'expansion_boost': 0.88,
                'description': '短模糊-中等召回+强扩散'
            })
        
        # 策略3: 短具体查询（4-8字，很具体）
        elif features.length <= 8 and features.is_specific:
            strategy.update({
                'recall_k': 15,
                'max_expanded': 20,
                'expansion_boost': 0.8,
                'description': '短具体-多召回+弱扩散'
            })
        
        # 策略4: 中等长度（9-20字）
        elif features.length <= 20:
            if features.keyword_density > 0.5:
                # 关键词密集
                strategy.update({
                    'recall_k': 12,
                    'max_expanded': 25,
                    'description': '中等密集-平衡策略'
                })
            else:
                # 关键词稀疏
                strategy.update({
                    'recall_k': 8,
                    'max_expanded': 35,
                    'expansion_boost': 0.9,
                    'description': '中等稀疏-少召回+强扩散'
                })
        
        # 策略5: 长查询（>20字）- 主要靠TF-IDF
        else:
            strategy.update({
                'recall_k': 20,
                'max_expanded': 15,
                'expansion_boost': 0.75,
                'description': '长查询-多召回+轻扩散'
            })
        
        return strategy
    
    def _semantic_recall(self, query: str, candidates: List[MemoryNode], top_k: int) -> List[Tuple[MemoryNode, float]]:
        """
        语义召回：用语义相似度召回候选
        
        Returns:
            [(node, semantic_score), ...]
        """
        query_emb = self.semantic_mgr.encode_single(query)
        
        scored = []
        for node in candidates:
            node_emb = self.semantic_mgr.load_embedding(node.id)
            if node_emb is None:
                node_emb = self.semantic_mgr.encode_single(node.content)
            
            doc_text = f"{node.title or ''} {node.content or ''}"
            sim = self.semantic_mgr.compute_similarity(
                query_emb, node_emb, query_text=query, doc_text=doc_text
            )
            scored.append((node, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def _multi_path_recall(
        self, 
        query: str, 
        candidates: List[MemoryNode],
        tfidf_k: int,
        semantic_k: int
    ) -> List[Tuple[MemoryNode, float, str]]:
        """
        多路召回：同时用 TF-IDF 和语义召回，融合去重
        
        Returns:
            [(node, score, source), ...] source为'tfidf'或'simantic'
        """
        # TF-IDF 召回
        self._build_tfidf_index(candidates)
        query_vec = self.tfidf_mgr.encode_single(query)
        
        tfidf_results = []
        for node in candidates:
            node_vec = self._node_embeddings.get(node.id)
            if node_vec is not None:
                norm_q = np.linalg.norm(query_vec)
                norm_n = np.linalg.norm(node_vec)
                if norm_q > 0 and norm_n > 0:
                    sim = np.dot(query_vec, node_vec) / (norm_q * norm_n)
                    sim = (sim + 1) / 2
                else:
                    sim = 0.0
            else:
                sim = 0.0
            
            # 关键词增强
            keyword_score = self._fast_keyword_score(query, node)
            final_sim = max(sim, keyword_score * 0.5)
            tfidf_results.append((node, final_sim))
        
        tfidf_results.sort(key=lambda x: x[1], reverse=True)
        tfidf_results = tfidf_results[:tfidf_k]
        
        # 语义召回
        semantic_results = self._semantic_recall(query, candidates, semantic_k)
        
        # 融合去重（加权平均）
        node_scores = {}  # node -> {'tfidf': score, 'semantic': score}
        
        for node, score in tfidf_results:
            if node.url not in node_scores:
                node_scores[node.url] = {'node': node, 'tfidf': score, 'semantic': 0}
            else:
                node_scores[node.url]['tfidf'] = score
        
        for node, score in semantic_results:
            if node.url not in node_scores:
                node_scores[node.url] = {'node': node, 'tfidf': 0, 'semantic': score}
            else:
                node_scores[node.url]['semantic'] = score
        
        # 计算融合分数
        fused_results = []
        for url, data in node_scores.items():
            # 两路都有的，加权融合；只有一路的，分数打折
            if data['tfidf'] > 0 and data['semantic'] > 0:
                fused_score = 0.4 * data['tfidf'] + 0.6 * data['semantic']
                source = 'both'
            elif data['tfidf'] > 0:
                fused_score = data['tfidf'] * 0.8  # 只有TF-IDF，打折
                source = 'tfidf'
            else:
                fused_score = data['semantic'] * 0.9  # 只有语义，轻微打折
                source = 'semantic'
            
            fused_results.append((data['node'], fused_score, source))
        
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:max(tfidf_k, semantic_k)]
    
    def adaptive_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        自适应检索 - 根据查询特征动态调整策略
        
        流程：
        1. 分析查询特征
        2. 选择最优策略
        3. 多路召回（可选）
        4. 子图扩散（可选）
        5. 语义精排
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            排序后的搜索结果
        """
        print(f"\n{'='*60}")
        print(f"🔍 自适应检索: '{query}'")
        print(f"{'='*60}")
        
        # 步骤1: 分析查询
        features = self._analyze_query(query)
        print(f"📊 查询特征: {features}")
        
        # 步骤2: 决策策略
        strategy = self._decide_strategy(features)
        print(f"🎯 策略选择: {strategy['description']}")
        print(f"   参数: recall_k={strategy['recall_k']}, "
              f"expanded={strategy['max_expanded']}, "
              f"boost={strategy['expansion_boost']}")
        
        # 步骤3: 召回阶段
        candidates = self.storage.get_all()
        if not candidates:
            return []
        
        if strategy['use_tfidf'] and strategy['use_semantic_seed']:
            # 多路召回
            print(f"🚀 多路召回: TF-IDF({strategy['recall_k']}) + 语义({strategy['recall_k']//2})")
            recalled = self._multi_path_recall(
                query, candidates, 
                tfidf_k=strategy['recall_k'],
                semantic_k=strategy['recall_k'] // 2
            )
            recall_nodes = [(n, s) for n, s, _ in recalled]
            recall_urls = {n.url for n, _ in recall_nodes}
        elif strategy['use_semantic_seed']:
            # 纯语义召回
            print(f"🚀 语义召回: top-{strategy['recall_k']}")
            recall_nodes = self._semantic_recall(query, candidates, strategy['recall_k'])
            recall_urls = {n.url for n, _ in recall_nodes}
        else:
            # TF-IDF召回
            print(f"🚀 TF-IDF召回: top-{strategy['recall_k']}")
            recall_nodes = self._first_stage_recall(query, candidates, strategy['recall_k'])
            recall_urls = {n.url for n, _ in recall_nodes}
        
        print(f"   召回 {len(recall_nodes)} 个候选")
        
        # 步骤4: 子图扩散（可选）
        expanded_nodes = {}
        if strategy['max_expanded'] > 0:
            print(f"🕸️  子图扩散: 深度={strategy['expansion_depth']}, 最大={strategy['max_expanded']}")
            
            for depth in range(strategy['expansion_depth']):
                current_layer = [n for n, _ in recall_nodes] if depth == 0 else list(expanded_nodes.values())
                new_count = 0
                
                for node in current_layer:
                    all_links = list(node.links) + list(node.backlinks)
                    
                    for link_url in all_links:
                        if link_url in recall_urls or link_url in expanded_nodes:
                            continue
                        if len(expanded_nodes) >= strategy['max_expanded']:
                            break
                        
                        linked_node = self.storage.load_by_url(link_url)
                        if linked_node:
                            expanded_nodes[link_url] = linked_node
                            new_count += 1
                    
                    if len(expanded_nodes) >= strategy['max_expanded']:
                        break
                
                print(f"   [深度{depth+1}] 新增 {new_count} 个节点")
                if new_count == 0:
                    break
            
            print(f"   扩散完成: 召回{len(recall_nodes)} + 扩展{len(expanded_nodes)}")
        
        # 步骤5: 语义精排
        print(f"🎨 语义精排...")
        all_nodes = [n for n, _ in recall_nodes] + list(expanded_nodes.values())
        all_pageranks = [n.pagerank for n in all_nodes]
        
        query_emb = self.semantic_mgr.encode_single(query)
        results = []
        
        # 精排召回节点
        for node, recall_score in recall_nodes:
            node_emb = self.semantic_mgr.load_embedding(node.id)
            if node_emb is None:
                node_emb = self.semantic_mgr.encode_single(node.content)
            
            doc_text = f"{node.title or ''} {node.content or ''}"
            semantic_score = self.semantic_mgr.compute_similarity(
                query_emb, node_emb, query_text=query, doc_text=doc_text
            )
            
            # 关键词提升
            keyword_score = self._fast_keyword_score(query, node)
            exact_match = self._is_exact_keyword_match(query, node)
            if exact_match or keyword_score >= 0.8:
                semantic_score = max(semantic_score, 0.7)
            elif keyword_score >= 0.5:
                semantic_score = max(semantic_score, 0.5)
            
            # PageRank
            max_pr = max(all_pageranks) if all_pageranks else 1.0
            pagerank_score = node.pagerank / max_pr if max_pr > 0 else 1.0
            
            # 时效性
            days_old = (datetime.now() - node.created).days
            if days_old < 7:
                recency_penalty = 0.3 if keyword_score < 0.8 else 0.8
            elif days_old < 30:
                recency_penalty = 0.9
            elif days_old < 365:
                recency_penalty = 1.0
            else:
                recency_penalty = 0.9
            
            # 综合分数
            base_score = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_PAGERANK * pagerank_score +
                0.1 * recall_score  # 召回分数小幅加权
            )
            final_score = base_score * recency_penalty
            
            results.append(SearchResult(
                node=node,
                semantic_score=semantic_score,
                pagerank_score=pagerank_score,
                recency_score=recency_penalty,
                final_score=min(final_score, 1.0),
                metadata={'source': 'recall', 'strategy': strategy['description']}
            ))
        
        # 精排扩展节点
        for url, node in expanded_nodes.items():
            node_emb = self.semantic_mgr.load_embedding(node.id)
            if node_emb is None:
                node_emb = self.semantic_mgr.encode_single(node.content)
            
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
                recency_penalty = 0.3 if keyword_score < 0.8 else 0.8
            elif days_old < 30:
                recency_penalty = 0.9
            elif days_old < 365:
                recency_penalty = 1.0
            else:
                recency_penalty = 0.9
            
            # 扩展节点打折
            base_score = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_PAGERANK * pagerank_score
            )
            final_score = base_score * recency_penalty * strategy['expansion_boost']
            
            results.append(SearchResult(
                node=node,
                semantic_score=semantic_score,
                pagerank_score=pagerank_score,
                recency_score=recency_penalty,
                final_score=min(final_score, 1.0),
                metadata={'source': 'expanded', 'strategy': strategy['description']}
            ))
        
        # 排序返回
        results.sort(key=lambda x: x.final_score, reverse=True)
        top_results = results[:top_k]
        
        recall_in_top = sum(1 for r in top_results if r.metadata.get('source') == 'recall')
        expanded_in_top = len(top_results) - recall_in_top
        
        print(f"✅ 完成: 返回 top-{top_k} (召回{recall_in_top}, 扩散{expanded_in_top})")
        print(f"{'='*60}\n")
        
        return top_results
