#!/usr/bin/env python3
"""
测试时效性修复效果
对比新旧记忆在查询中的表现
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.storage import MemoryStorage
from src.retrieval import MemoryRetrieval

def test_recency_fix():
    print("=" * 70)
    print("时效性修复效果测试")
    print("=" * 70)
    
    # Initialize
    data_dir = Path(__file__).parent / "data"
    storage = MemoryStorage(base_dir=str(data_dir))
    retrieval = MemoryRetrieval(storage)
    
    all_nodes = storage.get_all()
    print(f"\n📊 总节点数: {len(all_nodes)}")
    
    # 分析节点分布
    now = datetime.now()
    new_nodes = [n for n in all_nodes if (now - n.created).days < 1]  # 今天
    old_nodes = [n for n in all_nodes if (now - n.created).days > 7]  # 一周前
    
    print(f"\n📅 时间分布:")
    print(f"  - 今天创建: {len(new_nodes)} 节点")
    print(f"  - 一周前+: {len(old_nodes)} 节点")
    
    # 找出高 PageRank 的旧节点（核心知识）
    high_pr_threshold = 0.0025
    core_knowledge = [n for n in old_nodes if n.pagerank >= high_pr_threshold]
    
    print(f"\n🎯 核心知识节点 (旧但高PR):")
    for node in sorted(core_knowledge, key=lambda n: n.pagerank, reverse=True)[:5]:
        age_days = (now - node.created).days
        print(f"  - {node.title[:40] if node.title else node.id[:20]}... | PR: {node.pagerank:.4f} | {age_days}天前")
    
    # 测试查询
    test_cases = [
        # (查询词, 期望匹配核心知识)
        ("灵痕世界观", ["灵痕", "世界观", "设定"]),
        ("PageRank 修复", ["pagerank", "bug", "修复"]),
        ("auto-save hook", ["auto-save", "hook", "bug"]),
        ("memory system 架构", ["memory", "架构", "节点"]),
        ("检索时效性", ["时效性", "检索", "recency"]),
    ]
    
    print(f"\n🔍 测试查询 (看旧核心知识是否排在前面):")
    print("-" * 70)
    
    for query, expected_tags in test_cases:
        results = retrieval.search(query, top_k=5)
        
        print(f"\n查询: '{query}'")
        
        for i, r in enumerate(results[:3], 1):
            node = r.node
            age_days = (now - node.created).days
            is_old = age_days > 7
            is_high_pr = node.pagerank >= high_pr_threshold
            
            # 标记类型
            node_type = "🆕 新" if age_days < 1 else ("🎯 核心" if is_high_pr else "📄 普通")
            
            print(f"  {i}. {node.title[:35] if node.title else node.id[:20]}...")
            print(f"     {node_type} | PR: {node.pagerank:.4f} | {age_days}天前 | 得分: {r.final_score:.3f}")
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("✅ 如果看到 '🎯 核心' 节点排在前面 → 修复成功")
    print("✅ 如果 '得分' 高的是旧的高PR节点 → 修复成功")
    print("❌ 如果全是 '🆕 新' 节点排在前面 → 修复失败")

if __name__ == "__main__":
    test_recency_fix()
