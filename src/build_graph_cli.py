# -*- coding: utf-8 -*-
"""
Build Memory Graph CLI - Entry point for python -m src.build_graph_cli
"""
from .memory_system import MemorySystem


def main():
    print("🧠 重建记忆图谱...")
    print("-" * 40)
    
    # 初始化记忆系统
    ms = MemorySystem()
    
    # 获取统计信息
    stats = ms.stats()
    print(f"总节点数: {stats['total_nodes']}")
    print(f"总标签数: {stats['total_tags']}")
    print(f"平均 PageRank: {stats['avg_pagerank']:.4f}")
    print()
    
    # 构建图谱（自动建立链接 + 计算 PageRank）
    print("🔗 建立链接（语义相似度 + 时间窗口 + 共享标签）...")
    scores = ms.build_graph(auto_link=True)
    
    print()
    print("=" * 40)
    print(f"✅ 图谱重建完成！共 {len(scores)} 个节点")
    
    # 显示分布
    if scores:
        score_values = list(scores.values())
        print(f"\n📊 PageRank 分布:")
        print(f"   最小值: {min(score_values):.4f}")
        print(f"   最大值: {max(score_values):.4f}")
        print(f"   平均值: {sum(score_values)/len(score_values):.4f}")
        
        # Top 10
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Top 10 高分节点:")
        for i, (url, score) in enumerate(sorted_scores[:10], 1):
            node_id = url.split('/')[-1] if '/' in url else url
            print(f"   {i}. {node_id[:16]}...: {score:.4f}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
