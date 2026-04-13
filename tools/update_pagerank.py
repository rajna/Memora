#!/usr/bin/env python3
"""
更新 PageRank（自动先更新标签）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.storage import MemoryStorage
from src.pagerank import MemoryGraph
from src.tag_generator import TagGenerator
from src import config

def update_pagerank():
    print("=" * 70)
    print("更新标签 + PageRank")
    print("=" * 70)
    
    storage = MemoryStorage(base_dir=config.MEMORY_DIR)
    
    # 第1步：先更新标签
    print("\n📌 第1步：生成并应用标签...")
    print("-" * 50)
    try:
        generator = TagGenerator()
        result = generator.generate_tags(top_k=10)
        generator.apply_tags_to_nodes(result)
        print(f"✅ 标签更新完成！生成 {len(result.node_tags)} 个节点的标签")
        print(f"🔥 全局关键词 Top 5: {[w for w, s in result.global_keywords[:5]]}")
    except Exception as e:
        print(f"⚠️ 标签更新失败（继续PageRank）: {e}")
    
    # 第2步：更新 PageRank
    print("\n📌 第2步：计算并更新 PageRank...")
    print("-" * 50)
    
    all_nodes = storage.get_all()
    print(f"总节点: {len(all_nodes)}")
    
    # 构建图
    graph = MemoryGraph()
    graph.build_from_nodes(all_nodes)
    
    print(f"图中节点: {len(graph.graph.nodes())}")
    print(f"图中边: {len(graph.graph.edges())}")
    
    # 计算 PageRank
    print("\n计算 PageRank...")
    pagerank_scores = graph.calculate_pagerank()
    
    # 更新节点（PageRank返回的是url作为key）
    print("\n更新节点分数...")
    updated = 0
    for node in all_nodes:
        if node.url in pagerank_scores:
            node.pagerank = pagerank_scores[node.url]
            storage.save(node)
            updated += 1
    
    # 统计
    pr_values = list(pagerank_scores.values())
    print(f"\n✅ 更新完成!")
    print(f"  - 更新了 {updated}/{len(all_nodes)} 个节点")
    print(f"  - 最小 PR: {min(pr_values):.6f}")
    print(f"  - 最大 PR: {max(pr_values):.6f}")
    print(f"  - 平均 PR: {sum(pr_values)/len(pr_values):.6f}")
    
    # 检查高PR节点
    high_pr = [n for n in all_nodes if n.pagerank >= 0.9]
    print(f"  - PR≥0.9 节点: {len(high_pr)} (应为0)")

if __name__ == "__main__":
    update_pagerank()
