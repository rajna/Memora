#!/usr/bin/env python3
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.storage import MemoryStorage
from src.retrieval import TwoStageRetriever

with open('benchmark/test_2026_04_10_30.json') as f:
    test_data = json.load(f)
questions = test_data['questions']

storage = MemoryStorage(base_dir='data')
retriever = TwoStageRetriever(storage)

all_nodes = storage.get_all()
all_node_ids = {n.id for n in all_nodes}
valid = [q for q in questions if q.get('answer_id') in all_node_ids]
print(f'Total: {len(questions)}, Valid: {len(valid)}', flush=True)

results = {'r1': 0, 'r5': 0, 'r10': 0, 'details': []}
for i, q in enumerate(valid, 1):
    query = q['question']
    expected = q['answer_id']
    
    search_results = retriever.search_with_graph_expansion(
        query=query, top_k=10, recall_k=10, expansion_depth=1, max_expanded=50
    )
    
    retrieved = [r.node.id for r in search_results]
    rank = next((j+1 for j, rid in enumerate(retrieved) if rid == expected), -1)
    
    if rank == 1: results['r1'] += 1
    if rank <= 5: results['r5'] += 1
    if rank <= 10: results['r10'] += 1
    
    results['details'].append({'query': query[:50], 'rank': rank})
    
    status = '✅' if rank == 1 else ('🟡' if rank <= 5 else '❌')
    print(f'{status} [{i:2d}/{len(valid)}] R{rank:3d} | {query[:40]}...', flush=True)

total = len(valid)
print()
print('=' * 50)
print('Hybrid Strategy Results')
print('=' * 50)
print(f'Recall@1:  {results["r1"]:2d}/{total} = {results["r1"]/total*100:.1f}%')
print(f'Recall@5:  {results["r5"]:2d}/{total} = {results["r5"]/total*100:.1f}%')
print(f'Recall@10: {results["r10"]:2d}/{total} = {results["r10"]/total*100:.1f}%')

mrr = sum(1/d['rank'] for d in results['details'] if d['rank'] > 0) / total
print(f'MRR:       {mrr:.3f}')
