#!/usr/bin/env python3
"""
测试核心知识检索
验证旧的核心知识（高PR节点）能否被正确检索
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.storage import MemoryStorage
from src.retrieval import MemoryRetrieval

def test_core_knowledge():
    print("=" * 70)
    print("核心知识检索测试")
    print("=" * 70)
    
    storage = MemoryStorage(base_dir='./data')
    retrieval = MemoryRetrieval(storage)
    
    nodes = storage.get_all()
    now = datetime.now()
    
    # 找出高 PR 的旧节点（核心知识）
    high_pr_threshold = 0.003
    core_knowledge = [
        n for n in nodes 
        if n.pagerank >= high_pr_threshold and (now - n.created).days > 1
    ]
    
    print(f"\n🎯 核心知识节点 (PR>={high_pr_threshold}, 1天+):")
    print(f"共 {len(core_knowledge)} 个\n")
    
    for node in sorted(core_knowledge, key=lambda n: n.pagerank, reverse=True)[:8]:
        age = (now - node.created).days
        title = (node.title or "")[:50]
        print(f"  PR:{node.pagerank:.4f} | {age}天前 | {title}...")
    
    # 测试查询（基于核心知识的关键词）
    test_queries = [
        ("灵痕世界观", ["灵痕", "世界观", "沈灵"]),
        ("PageRank bug 修复", ["pagerank", "bug", "链接", "修复"]),
        ("memory system 架构", ["memory", "架构", "节点", "检索"]),
        ("TF-IDF 标签生成", ["tf-idf", "标签", "生成"]),
        ("auto-save 修复", ["auto-save", "hook", "bug", "修复"]),
        ("self-harness 引擎", ["self-harness", "引擎", "进化"]),
        ("MiniMax API", ["minimax", "api", "图片"]),
        ("修仙云服务项目", ["修仙", "云服务", "项目"]),
    ]
    
    print(f"\n🔍 测试查询（看核心知识是否排在前面）:")
    print("-" * 70)
    
    success_count = 0
    for query, expected_keywords in test_queries:
        results = retrieval.search(query, top_k=5)
        
        print(f"\n查询: '{query}'")
        
        # 检查前3个结果是否包含核心知识
        found_core = False
        for i, r in enumerate(results[:3], 1):
            node = r.node
            age = (now - node.created).days
            is_core = node.pagerank >= high_pr_threshold and age > 1
            
            # 检查是否包含关键词
            content_lower = (node.title or "" + " " + node.content).lower()
            has_keyword = any(kw.lower() in content_lower for kw in expected_keywords)
            
            marker = "✅" if is_core else "  "
            keyword_marker = "🎯" if has_keyword else "  "
            
            print(f"  {marker}{keyword_marker} {i}. {node.title[:40] if node.title else node.id[:20]}...")
            print(f"       PR:{node.pagerank:.4f} | {age}天前 | 得分:{r.final_score:.3f}")
            
            if is_core and has_keyword:
                found_core = True
        
        if found_core:
            success_count += 1
            print(f"   ✓ 找到核心知识!")
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"找到核心知识的查询: {success_count}/{len(test_queries)}")
    
    if success_count >= len(test_queries) * 0.5:
        print("✅ 修复成功 - 核心知识能被正确检索")
    else:
        print("❌ 需要优化 - 核心知识检索不够准确")

if __name__ == "__main__":
    test_core_knowledge()
