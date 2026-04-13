# -*- coding: utf-8 -*-
"""
Build All - One-step rebuild for tags + graph + PageRank
一键完成：标签生成 → 建图 → PageRank 计算

Usage:
    python -m tools.build_all
    python -m tools.build_all --skip-tags     # 只重建图谱
    python -m tools.build_all --skip-graph    # 只生成标签
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from pathlib import Path
from collections import Counter

from src.memory_system import MemorySystem
from src.storage import MemoryStorage
from src.embeddings import get_embedding_manager
from src.pagerank import MemoryGraph
from src.config import MEMORY_DIR, PAGERANK_DAMPING


def main():
    import argparse
    parser = argparse.ArgumentParser(description='一键重建：标签 + 图谱 + PageRank')
    parser.add_argument('--skip-tags', action='store_true', help='跳过标签生成')
    parser.add_argument('--skip-graph', action='store_true', help='跳过图谱构建')
    parser.add_argument('--top-k-tags', type=int, default=5, help='每个节点的标签数')
    parser.add_argument('--link-top-k', type=int, default=5, help='每个节点的链接数')
    parser.add_argument('--sim-threshold', type=float, default=0.8, help='相似度阈值')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 Memora 一键重建")
    print("=" * 60)
    print(f"  跳过标签: {args.skip_tags}")
    print(f"  跳过图谱: {args.skip_graph}")
    print(f"  标签数: {args.top_k_tags}")
    print(f"  链接数: {args.link_top_k}")
    print()
    
    # 初始化
    storage = MemoryStorage(MEMORY_DIR)
    embedding_mgr = get_embedding_manager()
    graph = MemoryGraph()
    
    # 加载所有节点
    nodes = list(storage.iterate_all())
    print(f"📚 加载了 {len(nodes)} 个记忆节点\n")
    
    if len(nodes) == 0:
        print("⚠️ 没有找到记忆节点")
        return 0
    
    # ========================================
    # Step 1: 生成标签 (可选)
    # ========================================
    if not args.skip_tags:
        print("🏷️  Step 1: 生成标签...")
        try:
            import jieba.analyse
            JIEBA_AVAILABLE = True
        except ImportError:
            JIEBA_AVAILABLE = False
            print("  ⚠️ jieba 未安装，跳过 TextRank 提取")
        
        # 停用词
        STOPWORDS = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
                     '都', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
                     '会', '着', '没有', '看', '好', '自己', '这', '那', '之', '与',
                     '及', '等', '或', '但', '而', '为', '于', '以', '被', '将'}
        
        tag_counts = Counter()
        tagged = 0
        
        for i, node in enumerate(nodes):
            if i % 50 == 0:
                print(f"  处理进度: {i}/{len(nodes)}")
            
            # 提取内容
            text = f"{node.title or ''} {node.content[:3000]}"
            if not text.strip():
                continue
            
            tags = []
            
            if JIEBA_AVAILABLE:
                try:
                    keywords = jieba.analyse.textrank(
                        text, topK=args.top_k_tags * 2,
                        withWeight=True,
                        allowPOS=('n', 'v', 'vn', 'nz', 'eng')
                    )
                    for word, weight in keywords:
                        word = word.lower().strip()
                        if (len(word) >= 2 and 
                            word not in STOPWORDS and
                            not word.isdigit() and
                            word not in tags):
                            tags.append(word)
                            tag_counts[word] += 1
                        if len(tags) >= args.top_k_tags:
                            break
                except:
                    pass
            
            # 更新节点标签
            if tags:
                node.tags = tags
                storage.save(node)
                tagged += 1
        
        print(f"  ✅ 标签生成完成！更新了 {tagged} 个节点")
        print(f"  🔥 Top 10 标签: {tag_counts.most_common(10)}")
        print()
    
    # ========================================
    # Step 2: 构建图谱 + 计算 PageRank
    # ========================================
    if not args.skip_graph:
        print("🔗 Step 2: 构建图谱 + PageRank...")
        
        # 添加所有节点
        for node in nodes:
            graph.add_node(node)
        
        # 预计算向量
        print("  计算向量相似度...")
        url_to_emb = {}
        for node in nodes:
            if node.embedding_file:
                emb = embedding_mgr.load_embedding(node.id)
                if emb is not None:
                    url_to_emb[node.url] = emb
        
        # 按时间排序
        nodes_sorted = sorted(nodes, key=lambda n: n.created)
        
        # 2.1 语义相似度链接
        print(f"  建立语义相似度链接...")
        for i, node1 in enumerate(nodes):
            if i % 50 == 0:
                print(f"    进度: {i}/{len(nodes)}")
            
            if node1.url not in url_to_emb:
                continue
            
            emb1 = url_to_emb[node1.url]
            similarities = []
            
            for node2 in nodes:
                if node2.url == node1.url or node2.url not in url_to_emb:
                    continue
                emb2 = url_to_emb[node2.url]
                sim = embedding_mgr.compute_similarity(emb1, emb2)
                if sim >= args.sim_threshold:
                    similarities.append((node2.url, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            for node2_url, sim in similarities[:args.link_top_k]:
                graph.add_edge(node1.url, node2_url, weight=sim)
                if node2_url not in node1.links:
                    node1.links.append(node2_url)
        
        # 2.2 时间相邻链接
        print("  建立时间相邻链接...")
        for i in range(len(nodes_sorted) - 1):
            node1, node2 = nodes_sorted[i], nodes_sorted[i + 1]
            if node2.url not in node1.links:
                graph.add_edge(node1.url, node2.url, weight=0.5)
                node1.links.append(node2.url)
            if node1.url not in node2.links:
                graph.add_edge(node2.url, node1.url, weight=0.5)
                node2.links.append(node1.url)
        
        # 2.3 共享标签链接
        print("  建立共享标签链接...")
        for node1 in nodes:
            tag_weights = []
            for node2 in nodes:
                if node2.url == node1.url:
                    continue
                shared = set(node1.tags) & set(node2.tags)
                if shared:
                    weight = len(shared) / max(len(node1.tags), len(node2.tags), 1)
                    tag_weights.append((node2.url, weight))
            
            tag_weights.sort(key=lambda x: x[1], reverse=True)
            for node2_url, weight in tag_weights[:args.link_top_k]:
                if node2_url not in node1.links:
                    graph.add_edge(node1.url, node2_url, weight=weight)
                    node1.links.append(node2_url)
        
        # 2.4 保存节点 + 更新 backlinks
        print("  保存节点...")
        for node in nodes:
            storage.save(node)
            # 更新被链接节点的 backlinks
            for target_url in node.links:
                target = storage.load_by_url(target_url)
                if target and node.url not in target.backlinks:
                    target.backlinks.append(node.url)
                    storage.save(target)
        
        # 2.5 计算 PageRank
        print("  计算 PageRank...")
        if len(graph.graph) > 0:
            import networkx as nx
            scores = nx.pagerank(
                graph.graph,
                alpha=PAGERANK_DAMPING,
                weight='weight'
            )
            
            # 更新节点 PageRank
            for node in nodes:
                node.pagerank = scores.get(node.url, 0)
                storage.save(node)
            
            # 排序显示
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n  📊 图谱统计:")
            print(f"     节点: {len(graph.graph.nodes)}")
            print(f"     边: {len(graph.graph.edges)}")
            print(f"     PageRank 范围: {min(scores.values()):.4f} ~ {max(scores.values()):.4f}")
            
            print(f"\n  🏆 Top 10:")
            for i, (url, score) in enumerate(sorted_scores[:10], 1):
                node_id = url.split('/')[-1][:20]
                print(f"     {i:2d}. {node_id:20s} {score:.4f}")
            
            # 孤立点警告
            isolated = [n for n in graph.graph.nodes() if graph.graph.degree(n) == 0]
            if isolated:
                print(f"\n  ⚠️  {len(isolated)} 个孤立节点 (无链接):")
                for url in isolated[:5]:
                    print(f"       - {url.split('/')[-1][:30]}")
        else:
            print("  ⚠️ 图谱为空，无法计算 PageRank")
    
    print()
    print("=" * 60)
    print("✅ 全部完成！")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
