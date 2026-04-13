#!/usr/bin/env python3
"""Hybrid Benchmark v3.0 - 快速版本（无日志输出）"""

import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.storage import MemoryStorage
from src.retrieval import TwoStageRetriever
from src.config import MEMORY_DIR

def main():
    with open('benchmark/test_2026_04_10_30.json') as f:
        questions = json.load(f)['questions']

    print('Hybrid Benchmark v3.0 - 50题')
    
    storage = MemoryStorage(MEMORY_DIR)
    retriever = TwoStageRetriever(storage, first_stage_k=10)

    all_results = []
    start_all = time.time()

    for i, q in enumerate(questions):
        start = time.time()
        search_results = retriever.search(q['question'][:60], top_k=5)
        elapsed = time.time() - start
        found = any(r.node.id == q['answer_id'] for r in search_results)
        rank = next((j+1 for j, r in enumerate(search_results) if r.node.id == q['answer_id']), None)
        all_results.append({'hit': found, 'rank': rank, 'time': elapsed})

    total_time = time.time() - start_all

    # 统计
    hits = sum(1 for r in all_results if r['hit'])
    recall_1 = sum(1 for r in all_results if r['hit'] and r['rank']==1) / 50 * 100
    recall_5 = hits / 50 * 100
    mrr = sum(1/r['rank'] for r in all_results if r['hit'] and r['rank']) / 50

    print(f'Recall@1: {recall_1:.1f}% | Recall@5: {recall_5:.1f}% | MRR: {mrr:.3f}')
    print(f'Hit: {hits}/50 | Total: {total_time:.1f}s')

    # 保存
    result_data = {
        'version': '3.0', 'method': 'hybrid', 'total': 50,
        'hit': hits, 'recall_1': round(recall_1, 2), 'recall_5': round(recall_5, 2),
        'mrr': round(mrr, 3), 'total_time': round(total_time, 1),
        'results': all_results
    }
    with open('benchmark/hybrid_v3_results.json', 'w') as f:
        json.dump(result_data, f, indent=2)

if __name__ == '__main__':
    main()
