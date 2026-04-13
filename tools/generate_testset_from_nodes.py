#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 memory-nodes 生成测试集
- 2021-2024: 每50个节点取1个作为测试集 (LME格式: **User:**/**Assistant:**)
- 2026: 随机取30个 (auto-save格式: [用户]/[AI])
"""

import os
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any

MEMORY_NODES_DIR = Path("/Users/rama/.nanobot/workspace/Memora/data/memory-nodes")
YEAR_2026_DIR = Path("/Users/rama/.nanobot/workspace/Memora/data/2026")

def extract_qa_pairs_auto_save(content: str) -> List[Dict[str, str]]:
    """从 [用户]/[AI] 格式提取问答对"""
    qa_pairs = []
    # 匹配 [用户] 开头的内容
    pattern = r'\[用户\]\s*(.+?)(?=\n\[AI\]|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        question = match.strip()
        if len(question) > 5 and len(question) < 500:
            qa_pairs.append({"question": question})
    
    return qa_pairs

def extract_qa_pairs_lme(content: str) -> List[Dict[str, str]]:
    """从 **User:**/**Assistant:** 格式提取问答对"""
    qa_pairs = []
    # 匹配 **User:** 开头的内容
    pattern = r'\*\*User:\*\*\s*(.+?)(?=\n\*\*Assistant:\*\*|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        question = match.strip()
        if len(question) > 5 and len(question) < 500:
            qa_pairs.append({"question": question})
    
    return qa_pairs

def get_answer_auto_save(content: str, question: str) -> str:
    """获取 [AI] 回答"""
    pattern = rf'\[AI\]\s*(.+?)(?=\n\[用户\]|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    for m in matches:
        if len(m.strip()) > 0:
            return m.strip()[:300]
    return ""

def get_answer_lme(content: str, question: str) -> str:
    """获取 **Assistant:** 回答"""
    pattern = r'\*\*Assistant:\*\*\s*(.+?)(?=\n\*\*User:\*\*|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    for m in matches:
        if len(m.strip()) > 0:
            return m.strip()[:300]
    return ""

def collect_nodes_by_year():
    """收集各年份的节点"""
    nodes_by_year = {year: [] for year in [2021, 2022, 2023, 2024, 2026]}
    
    # 收集 2021-2024 (memory-nodes目录)
    for year in [2021, 2022, 2023, 2024]:
        year_dir = MEMORY_NODES_DIR / str(year)
        if year_dir.exists():
            md_files = sorted(year_dir.rglob("*.md"))
            nodes_by_year[year] = [str(f) for f in md_files]
            print(f"  {year}: {len(md_files)} nodes")
    
    # 收集 2026 (data/2026目录)
    if YEAR_2026_DIR.exists():
        md_files = sorted(YEAR_2026_DIR.rglob("*.md"))
        nodes_by_year[2026] = [str(f) for f in md_files]
        print(f"  2026: {len(md_files)} nodes")
    
    return nodes_by_year

def generate_testset(nodes_by_year: Dict[int, List[str]], target_2021_2024: int = 50) -> Dict[str, Any]:
    """生成测试集"""
    questions = []
    
    # 计算 2021-2024 各年份应取数量（均匀分配 + 最小保证）
    years = [2021, 2022, 2023, 2024]
    total_nodes = sum(len(nodes_by_year[y]) for y in years)
    
    # 均匀分配：每个年份基础数量 = 50 / 4 = 12
    base = target_2021_2024 // len(years)  # 12
    remainder = target_2021_2024 % len(years)  # 2
    
    counts = {}
    for i, year in enumerate(years):
        # 每个年份至少1个，最多不超过节点数
        count = min(base + (1 if i < remainder else 0), len(nodes_by_year[year]))
        counts[year] = max(count, 1) if len(nodes_by_year[year]) > 0 else 0
    
    # 调整总数（如果某些年份节点太少）
    actual_total = sum(counts.values())
    if actual_total != target_2021_2024:
        # 从最大的年份补足或减少
        diff = target_2021_2024 - actual_total
        counts[2023] += diff
    
    print(f"\n2021-2024 共{target_2021_2024}题，分布: {counts}")
    
    # 2021-2024: 按比例均匀采样 (LME格式)
    for year in years:
        nodes = nodes_by_year[year]
        n_samples = counts[year]
        print(f"\n处理 {year} ({len(nodes)} nodes, 取{n_samples}个)...")
        
        if n_samples == 0:
            continue
        
        # 均匀采样索引
        if len(nodes) <= n_samples:
            sample_indices = list(range(len(nodes)))
        else:
            step = len(nodes) / n_samples
            sample_indices = [int(i * step) for i in range(n_samples)]
        
        count = 0
        for idx in sample_indices:
            node_path = nodes[idx]
            node_id = Path(node_path).stem
            
            try:
                with open(node_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取问答对 (LME格式)
                qa_pairs = extract_qa_pairs_lme(content)
                if qa_pairs:
                    qa = qa_pairs[0]
                    question = qa['question']
                    answer = get_answer_lme(content, question)
                    
                    questions.append({
                        "question": question,
                        "answer": answer or node_id,
                        "answer_id": node_id,
                        "type": "memory_node",
                        "year": year,
                        "node_index": idx
                    })
                    count += 1
                    if count <= 3:
                        print(f"    [{idx}/{len(nodes)}] Q: {question[:60]}...")
                else:
                    # 如果没有问答对，使用标题
                    title_match = re.search(r'title:\s*(.+)', content)
                    title = title_match.group(1).strip() if title_match else node_id
                    questions.append({
                        "question": f"关于: {title}",
                        "answer": node_id,
                        "answer_id": node_id,
                        "type": "memory_node",
                        "year": year,
                        "node_index": idx
                    })
                    count += 1
                    if count <= 3:
                        print(f"    [{idx}/{len(nodes)}] (title only)")
                    
            except Exception as e:
                print(f"    Error: {e}")
        
        print(f"  -> 生成 {count} 个测试")
    
    # 2026: 随机取30个 (auto-save格式)
    nodes_2026 = nodes_by_year[2026]
    print(f"\n处理 2026 ({len(nodes_2026)} nodes, 随机取30个)...")
    
    if len(nodes_2026) >= 30:
        sample_nodes = random.sample(nodes_2026, 30)
    else:
        sample_nodes = nodes_2026
        print(f"  警告: 2026只有{len(nodes_2026)}个节点，不足30个")
    
    for idx, node_path in enumerate(sample_nodes):
        node_id = Path(node_path).stem
        
        try:
            with open(node_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取问答对 (auto-save格式)
            qa_pairs = extract_qa_pairs_auto_save(content)
            if qa_pairs:
                qa = qa_pairs[0]
                question = qa['question']
                answer = get_answer_auto_save(content, question)
                
                questions.append({
                    "question": question,
                    "answer": answer or node_id,
                    "answer_id": node_id,
                    "type": "memory_node",
                    "year": 2026
                })
                if idx < 3:
                    print(f"    Q: {question[:60]}...")
            else:
                title_match = re.search(r'title:\s*(.+)', content)
                title = title_match.group(1).strip() if title_match else node_id
                questions.append({
                    "question": f"关于: {title}",
                    "answer": node_id,
                    "answer_id": node_id,
                    "type": "memory_node",
                    "year": 2026
                })
                if idx < 3:
                    print(f"    (title only)")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    return {
        "description": "从 memory-nodes 自动生成的测试集 (2021-2024每50取1, 2026随机30)",
        "created": "2026-04-11T01:45:00",
        "total_questions": len(questions),
        "source": {
            "2021": f"每50取1, 约{len(nodes_by_year[2021])//50 + 1}题",
            "2022": f"每50取1, 约{len(nodes_by_year[2022])//50 + 1}题",
            "2023": f"每50取1, 约{len(nodes_by_year[2023])//50 + 1}题",
            "2024": f"每50取1, 约{len(nodes_by_year[2024])//50 + 1}题",
            "2026": f"随机{min(30, len(nodes_by_year[2026]))}题"
        },
        "questions": questions
    }

def main():
    print("=" * 60)
    print("测试集生成器 - 从 memory-nodes")
    print("=" * 60)
    
    random.seed(42)
    
    print("\n收集节点...")
    nodes_by_year = collect_nodes_by_year()
    
    print("\n生成测试集...")
    testset = generate_testset(nodes_by_year)
    
    # 保存
    output_path = Path("/Users/rama/.nanobot/workspace/Memora/benchmark/longmemeval_test_80.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 已保存到: {output_path}")
    print(f"✓ 总计 {len(testset['questions'])} 个测试问题")
    
    # 统计
    year_counts = {}
    for q in testset['questions']:
        year = q.get('year', 'unknown')
        year_counts[year] = year_counts.get(year, 0) + 1
    
    print("\n各年份分布:")
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count}")

if __name__ == "__main__":
    main()
