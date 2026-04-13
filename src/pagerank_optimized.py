"""
Optimized PageRank with Hub Dampening
优化版 PageRank - 抑制 Hub 节点垄断
"""
import networkx as nx
import numpy as np
from typing import List, Dict
from .models import MemoryNode
from .config import PAGERANK_DAMPING, PAGERANK_ITERATIONS, PAGERANK_TOLERANCE


class OptimizedMemoryGraph:
    """
    优化版记忆图谱
    - Hub 节点惩罚
    - 内容质量加权
    """
    
    def __init__(self, hub_penalty: float = 0.3, max_out_degree: int = 50):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, MemoryNode] = {}
        self.hub_penalty = hub_penalty  # Hub 惩罚系数
        self.max_out_degree = max_out_degree  # 最大出链数限制
    
    def add_node(self, node: MemoryNode):
        self.nodes[node.url] = node
        self.graph.add_node(node.url, node=node)
    
    def add_edge(self, from_url: str, to_url: str, weight: float = 1.0):
        if from_url in self.nodes and to_url in self.nodes:
            self.graph.add_edge(from_url, to_url, weight=weight)
    
    def _apply_hub_penalty(self, pagerank: Dict[str, float]) -> Dict[str, float]:
        """
        应用 Hub 惩罚：出链过多的节点降低权重
        """
        adjusted = {}
        for url, score in pagerank.items():
            out_degree = self.graph.out_degree(url)
            
            if out_degree > self.max_out_degree:
                # 超出阈值部分按比例惩罚
                penalty = 1.0 - self.hub_penalty * (out_degree / self.max_out_degree)
                penalty = max(0.1, min(1.0, penalty))  # 限制在 0.1-1.0
                adjusted[url] = score * penalty
            else:
                adjusted[url] = score
        
        # 重新归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def calculate_pagerank(self, apply_hub_penalty: bool = True) -> Dict[str, float]:
        """计算优化版 PageRank"""
        if len(self.graph) == 0:
            return {}
        
        # 基础 PageRank
        pagerank = nx.pagerank(
            self.graph,
            alpha=PAGERANK_DAMPING,
            max_iter=PAGERANK_ITERATIONS,
            tol=PAGERANK_TOLERANCE,
            weight='weight'
        )
        
        # 应用 Hub 惩罚
        if apply_hub_penalty:
            pagerank = self._apply_hub_penalty(pagerank)
        
        return pagerank
    
    def build_optimized(self, nodes: List[MemoryNode], 
                        similarity_threshold: float = 0.75):
        """
        构建优化版图谱
        - 限制每个节点的出链数
        - 优先保留高质量链接
        """
        self.graph.clear()
        self.nodes.clear()
        
        for node in nodes:
            self.add_node(node)
        
        # 只保留显式链接（不自动添加时间/标签链接）
        for node in nodes:
            # 限制出链数，保留 PR 最高的目标
            sorted_links = sorted(
                node.links, 
                key=lambda url: self.nodes.get(url, MemoryNode(id="", content="")).pagerank 
                if url in self.nodes else 0,
                reverse=True
            )[:self.max_out_degree]
            
            for target_url in sorted_links:
                if target_url in self.nodes:
                    self.add_edge(node.url, target_url)
        
        return self


def build_optimized_graph(nodes: List[MemoryNode]) -> OptimizedMemoryGraph:
    """便捷函数：构建优化版图谱"""
    graph = OptimizedMemoryGraph()
    graph.build_optimized(nodes)
    return graph
