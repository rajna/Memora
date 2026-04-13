#!/usr/bin/env python3
"""
调试 PageRank 状态
"""
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.storage import MemoryStorage
from src.pagerank import MemoryGraph

def debug_pagerank():
    print("=" * 70)
    print("PageRank 状态诊断")
    print("=" * 70)
    
    data_dir = Path(__file__).parent / "data"
    storage = MemoryStorage(base_dir=str(data_dir))
    
    all_nodes = storage.get_all()
    print(f"\n📊 总节点数: {len(all_nodes)}")
    
    # PageRank 分布
    pr_values = [n.pagerank for n in all_nodes]
    print(f"\n📈 PageRank 统计:")
    print(f"  - 最小值: {min(pr_values):.6f}")
    print(f"  - 最大值: {max(pr_values):.6f}")
    print(f"  - 平均值: {sum(pr_values)/len(pr_values):.6f}")
    print(f"  - PR=1.0 的节点数: {sum(1 for p in pr_values if p >= 0.999)}")
    
    # 时间分布
    now = datetime.now()
    age_buckets = Counter()
    for n in all_nodes:
        days = (now - n.created).days
        if days == 0:
            age_buckets["今天"] += 1
        elif days <= 7:
            age_buckets["1-7天"] += 1
        elif days <= 30:
            age_buckets["1个月内"] += 1
        else:
            age_buckets["更早"] += 1
    
    print(f"\n📅 时间分布:")
    for bucket, count in sorted(age_buckets.items()):
        print(f"  - {bucket}: {count} 节点")
    
    # 链接统计
    total_links = sum(len(n.links) for n in all_nodes)
    total_backlinks = sum(len(n.backlinks) for n in all_nodes)
    print(f"\n🔗 链接统计:")
    print(f"  - 总出链: {total_links}")
    print(f"  - 总反链: {total_backlinks}")
    
    # 检查高 PR 节点
    print(f"\n🎯 高 PageRank 节点 (Top 5):")
    for n in sorted(all_nodes, key=lambda x: x.pagerank, reverse=True)[:5]:
        days = (now - n.created).days
        print(f"  - PR: {n.pagerank:.4f} | {days}天前 | {n.title[:40] if n.title else n.id[:20]}...")
    
    # 检查低 PR 节点
    print(f"\n📄 低 PageRank 节点 (Bottom 5):")
    for n in sorted(all_nodes, key=lambda x: x.pagerank)[:5]:
        days = (now - n.created).days
        print(f"  - PR: {n.pagerank:.4f} | {days}天前 | {n.title[:40] if n.title else n.id[:20]}...")
    
    # 检查是否有链接的节点
    nodes_with_links = [n for n in all_nodes if n.links]
    print(f"\n🔗 有出链的节点: {len(nodes_with_links)}")
    if nodes_with_links:
        print("  示例:")
        for n in nodes_with_links[:3]:
            print(f"    - {n.title[:30] if n.title else n.id[:20]}... → {len(n.links)} 链接")

if __name__ == "__main__":
    debug_pagerank()
