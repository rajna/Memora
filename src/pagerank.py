"""
PageRank Algorithm for Memory Graph
记忆图谱的PageRank算法实现
"""
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

from .models import MemoryNode
from .config import PAGERANK_DAMPING, PAGERANK_ITERATIONS, PAGERANK_TOLERANCE


class MemoryGraph:
    """
    记忆图谱 - 管理节点间的链接关系
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # 有向图
        self.nodes: Dict[str, MemoryNode] = {}
    
    def add_node(self, node: MemoryNode):
        """添加节点到图谱"""
        self.nodes[node.url] = node
        self.graph.add_node(node.url, node=node)
    
    def add_edge(self, from_url: str, to_url: str, weight: float = 1.0, edge_type: str = "default"):
        """添加边（链接）
        
        Args:
            from_url: 源节点URL
            to_url: 目标节点URL
            weight: 边权重
            edge_type: 链接类型 (semantic/temporal/tag/default)
        """
        if from_url in self.nodes and to_url in self.nodes:
            self.graph.add_edge(from_url, to_url, weight=weight, edge_type=edge_type)
    
    def build_from_nodes(self, nodes: List[MemoryNode]):
        """从节点列表构建完整图谱"""
        self.graph.clear()
        self.nodes.clear()
        
        # 添加所有节点
        for node in nodes:
            self.add_node(node)
        
        # 添加边（从links字段）
        edge_count = 0
        for node in nodes:
            for target_url in node.links:
                if target_url in self.nodes:
                    self.add_edge(node.url, target_url)
                    edge_count += 1
        
        print(f"  [Graph] 节点: {len(self.nodes)}, 边: {edge_count}")
        return edge_count
    
    def calculate_pagerank(self) -> Dict[str, float]:
        """
        计算PageRank分数
        
        Returns:
            {url: pagerank_score}
        """
        if len(self.graph) == 0:
            return {}
        
        # 使用NetworkX的PageRank实现
        pagerank = nx.pagerank(
            self.graph,
            alpha=PAGERANK_DAMPING,
            max_iter=PAGERANK_ITERATIONS,
            tol=PAGERANK_TOLERANCE,
            weight='weight'
        )
        
        return pagerank
    
    def update_pagerank_scores(self):
        """更新所有节点的PageRank分数"""
        scores = self.calculate_pagerank()
        
        for url, score in scores.items():
            if url in self.nodes:
                self.nodes[url].pagerank = score
        
        return scores
    
    def get_backlinks(self, url: str) -> List[str]:
        """获取指向该URL的所有链接（反链）"""
        if url not in self.graph:
            return []
        return list(self.graph.predecessors(url))
    
    def get_outgoing_links(self, url: str) -> List[str]:
        """获取该URL指向的所有链接"""
        if url not in self.graph:
            return []
        return list(self.graph.successors(url))
    
    def find_similar_nodes_by_links(
        self, 
        url: str, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        通过链接相似度找到相关节点（协同过滤思想）
        
        Args:
            url: 目标节点URL
            top_k: 返回前k个
            
        Returns:
            [(similar_url, similarity_score), ...]
        """
        if url not in self.nodes:
            return []
        
        target_links = set(self.get_outgoing_links(url))
        target_backlinks = set(self.get_backlinks(url))
        
        similarities = []
        
        for other_url, other_node in self.nodes.items():
            if other_url == url:
                continue
            
            other_links = set(self.get_outgoing_links(other_url))
            other_backlinks = set(self.get_backlinks(other_url))
            
            # Jaccard相似度
            link_sim = len(target_links & other_links) / max(len(target_links | other_links), 1)
            backlink_sim = len(target_backlinks & other_backlinks) / max(len(target_backlinks | other_backlinks), 1)
            
            # 加权平均
            total_sim = (link_sim + backlink_sim) / 2
            similarities.append((other_url, total_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def auto_build_links(
        self, 
        nodes: List[MemoryNode],
        similarity_threshold: float = 0.8,
        top_k: int = 5
    ):
        """
        自动构建链接（每节点只取top-k）
        
        策略：
        1. 每个节点取语义相似度top-k → 建立链接
        2. 时间序列直接相邻 → 建立链接
        3. 每个节点取共享标签权重top-k → 建立链接
        """
        from .embeddings import get_embedding_manager
        
        # 首先添加所有节点到图谱
        self.graph.clear()
        self.nodes.clear()
        for node in nodes:
            self.add_node(node)
        
        print(f"  Added {len(nodes)} nodes to graph (top_k={top_k})")
        
        embedding_mgr = get_embedding_manager()
        
        # 预计算所有向量
        url_to_emb = {}
        for node in nodes:
            if node.embedding_file:
                emb = embedding_mgr.load_embedding(node.id)
                if emb is not None:
                    url_to_emb[node.url] = emb
        
        # 按时间排序
        nodes_sorted = sorted(nodes, key=lambda n: n.created)
        
        # 1. 基于语义相似度 - 每个节点取top-k
        print(f"  建立语义相似度链接 (top_k={top_k}, threshold={similarity_threshold})...")
        sem_links = 0
        for node1 in nodes:
            if node1.url not in url_to_emb:
                continue
            
            emb1 = url_to_emb[node1.url]
            
            # 计算与所有其他节点的相似度
            similarities = []
            for node2 in nodes:
                if node2.url == node1.url or node2.url not in url_to_emb:
                    continue
                emb2 = url_to_emb[node2.url]
                sim = embedding_mgr.compute_similarity(emb1, emb2)
                if sim >= similarity_threshold:
                    similarities.append((node2.url, sim))
            
            # 取top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            for node2_url, sim in similarities[:top_k]:
                self.add_edge(node1.url, node2_url, weight=sim, edge_type="semantic")
                if node2_url not in node1.links:
                    node1.links.append(node2_url)
                    sem_links += 1
        
        print(f"    语义链接: {sem_links} 条")
        
        # 2. 基于时间序列直接相邻建立链接
        print(f"  建立时间相邻链接...")
        temp_links = 0
        for i in range(len(nodes_sorted) - 1):
            node1 = nodes_sorted[i]
            node2 = nodes_sorted[i + 1]
            
            # 双向链接
            if node2.url not in node1.links:
                self.add_edge(node1.url, node2.url, weight=0.5, edge_type="temporal")
                node1.links.append(node2.url)
                temp_links += 1
            if node1.url not in node2.links:
                self.add_edge(node2.url, node1.url, weight=0.5, edge_type="temporal")
                node2.links.append(node1.url)
                temp_links += 1
        
        print(f"    时间相邻链接: {temp_links} 条")
        
        # 3. 基于共享tags - 每个节点取top-k
        print(f"  建立共享标签链接 (top_k={top_k})...")
        tag_links = 0
        for node1 in nodes:
            # 计算与所有其他节点的标签权重
            tag_weights = []
            for node2 in nodes:
                if node2.url == node1.url:
                    continue
                shared_tags = set(node1.tags) & set(node2.tags)
                if shared_tags:
                    weight = len(shared_tags) / max(len(node1.tags), len(node2.tags), 1)
                    tag_weights.append((node2.url, weight))
            
            # 取top-k
            tag_weights.sort(key=lambda x: x[1], reverse=True)
            for node2_url, weight in tag_weights[:top_k]:
                if node2_url not in node1.links:
                    self.add_edge(node1.url, node2_url, weight=weight, edge_type="tag")
                    node1.links.append(node2_url)
                    tag_links += 1
        
        print(f"    共享标签链接: {tag_links} 条")


def build_and_rank(nodes: List[MemoryNode]) -> MemoryGraph:
    """
    构建图谱并计算PageRank的便捷函数
    
    Returns:
        包含PageRank分数的MemoryGraph
    """
    graph = MemoryGraph()
    graph.build_from_nodes(nodes)
    graph.update_pagerank_scores()
    return graph
