#!/usr/bin/env python3
"""
Simple benchmark test for memory retrieval system
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.storage import MemoryStorage
from src.retrieval import MemoryRetrieval
from src.embeddings import get_embedding_manager

def test_retrieval():
    """Test basic retrieval functionality"""
    print("=" * 60)
    print("Memory Retrieval Benchmark Test")
    print("=" * 60)
    
    # Initialize
    data_dir = Path(__file__).parent / "data"
    storage = MemoryStorage(base_dir=str(data_dir))
    retrieval = MemoryRetrieval(storage)
    
    print(f"\n📊 System Status:")
    print(f"  - Total nodes: {len(storage.get_all())}")
    
    # Test queries based on recent memories
    test_queries = [
        "MemPalace 对比",
        "sentence-transformers 安装",
        "LongMemEval benchmark",
        "PageRank 图谱",
        "auto-save hook bug",
        "昨天的 memory 对比",
        "embedding 修复",
        "skill chain optimizer",
        "修仙云服务项目",
        "小说写作分析",
    ]
    
    print(f"\n🔍 Running {len(test_queries)} test queries...\n")
    
    results = []
    for query in test_queries:
        print(f"Query: '{query}'")
        search_results = retrieval.search(query, top_k=5)
        
        if search_results:
            top_result = search_results[0]
            print(f"  Top result: {top_result.node.title or top_result.node.id[:20]}...")
            print(f"  Score: {top_result.final_score:.3f} (semantic: {top_result.semantic_score:.3f}, pagerank: {top_result.pagerank_score:.3f})")
        else:
            print(f"  No results found!")
        print()
        
        results.append({
            "query": query,
            "found": len(search_results) > 0,
            "top_score": search_results[0].final_score if search_results else 0,
            "num_results": len(search_results)
        })
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    found_count = sum(1 for r in results if r["found"])
    print(f"  Queries with results: {found_count}/{len(test_queries)}")
    print(f"  Success rate: {found_count/len(test_queries)*100:.1f}%")
    
    return results

if __name__ == "__main__":
    test_retrieval()
