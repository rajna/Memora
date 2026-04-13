#!/usr/bin/env python3
"""构建 graph.pkl 供 benchmark 使用"""

import pickle
import yaml
from pathlib import Path
from collections import defaultdict


def extract_frontmatter(content: str) -> dict:
    """提取 YAML frontmatter"""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                return yaml.safe_load(parts[1]) or {}
            except:
                return {}
    return {}


def build_graph():
    """从节点文件构建图谱"""
    data_dir = Path("/Users/rama/.nanobot/workspace/Memora/data")
    index_dir = Path("/Users/rama/.nanobot/workspace/Memora/index")
    
    # 构建图结构
    nodes = {}
    edges = []
    
    for md_file in data_dir.rglob("*.md"):
        content = md_file.read_text(encoding='utf-8')
        fm = extract_frontmatter(content)
        
        node_id = fm.get('id') or md_file.stem
        
        # 节点属性
        nodes[node_id] = {
            'id': node_id,
            'title': fm.get('title', ''),
            'tags': fm.get('tags', []),
            'pagerank': 1.0 / 282,  # 初始均匀分布
        }
        
        # 从 links 提取边
        for link in fm.get('links', []):
            # 提取节点ID从路径
            if isinstance(link, str):
                link_id = link.split('/')[-1]
                edges.append((node_id, link_id))
            elif isinstance(link, dict):
                target = link.get('target', '')
                if target:
                    link_id = target.split('/')[-1]
                    edges.append((node_id, link_id))
    
    # 简单的 PageRank 计算
    damping = 0.85
    iterations = 20
    
    # 初始化
    n = len(nodes)
    if n == 0:
        print("⚠️  没有找到节点")
        return
    
    for node_id in nodes:
        nodes[node_id]['pagerank'] = 1.0 / n
    
    # 构建邻接表
    outgoing = defaultdict(list)
    for src, dst in edges:
        if src in nodes and dst in nodes:
            outgoing[src].append(dst)
    
    # 迭代计算
    for _ in range(iterations):
        new_ranks = {}
        for node_id in nodes:
            rank = (1 - damping) / n
            
            # 收集入链
            for src, dst in edges:
                if dst == node_id and src in outgoing:
                    rank += damping * nodes[src]['pagerank'] / len(outgoing[src])
            
            new_ranks[node_id] = rank
        
        # 更新
        for node_id, rank in new_ranks.items():
            nodes[node_id]['pagerank'] = rank
    
    # 保存
    graph = {
        'nodes': nodes,
        'edges': edges,
    }
    
    graph_file = index_dir / "graph.pkl"
    with open(graph_file, 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"✅ 构建图谱: {len(nodes)} 个节点, {len(edges)} 条边")
    
    # 显示 top 5 PageRank
    sorted_nodes = sorted(nodes.items(), key=lambda x: x[1]['pagerank'], reverse=True)
    print("\n🏆 Top 5 PageRank:")
    for node_id, data in sorted_nodes[:5]:
        print(f"   {node_id[:20]}...: {data['pagerank']:.6f}")


if __name__ == "__main__":
    build_graph()
