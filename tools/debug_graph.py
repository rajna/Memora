#!/usr/bin/env python3
"""
调试图结构
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.storage import MemoryStorage
from src.pagerank import MemoryGraph

def debug_graph():
    print("=" * 70)
    print("图结构诊断")
    print("=" * 70)
    
    data_dir = Path(__file__).parent / "data"
    storage = MemoryStorage(base_dir=str(data_dir))
    
    all_nodes = storage.get_all()
    print(f"总节点: {len(all_nodes)}")
    
    # 构建图
    graph = MemoryGraph()
    graph.build_from_nodes(all_nodes)
    
    print(f"\n📊 图统计:")
    print(f"  - 图中节点数: {len(graph.graph.nodes())}")
    print(f"  - 图中边数: {len(graph.graph.edges())}")
    print(f"  - 平均出度: {len(graph.graph.edges()) / len(graph.graph.nodes()):.1f}")
    
    # 检查 PR=1.0 的节点
    pr1_nodes = [n for n in all_nodes if n.pagerank >= 0.999]
    print(f"\n🔴 PR=1.0 的节点分析 ({len(pr1_nodes)} 个):")
    
    for node in pr1_nodes:
        # 在图中的入度
        in_degree = graph.graph.in_degree(node.id) if node.id in graph.graph else 0
        out_degree = graph.graph.out_degree(node.id) if node.id in graph.graph else 0
        
        print(f"\n  节点: {node.id[:20]}...")
        print(f"    - 入度: {in_degree}")
        print(f"    - 出度: {out_degree}")
        print(f"    - 存储的links: {len(node.links)}")
        print(f"    - 存储的backlinks: {len(node.backlinks)}")
        
        # 检查是否有入边
        if in_degree == 0:
            print(f"    ⚠️ 警告: 没有入边！")
    
    # 检查孤立节点
    isolated = [n for n in all_nodes if n.id in graph.graph and graph.graph.degree(n.id) == 0]
    print(f"\n⚠️ 孤立节点数: {len(isolated)}")

if __name__ == "__main__":
    debug_graph()
