#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 data/2026/04/10 节点生成30个测试集
- 格式: [用户]/[AI] (auto-save格式)
- 随机抽取30个节点
"""

import os
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any

SOURCE_DIR = Path("/Users/rama/.nanobot/workspace/Memora/data/2026/04/10")
OUTPUT_PATH = Path("/Users/rama/.nanobot/workspace/Memora/benchmark/test_2026_04_10_30.json")

def extract_qa_pairs(content: str) -> List[Dict[str, str]]:
    """从 [用户]/[AI] 格式提取问答对"""
    qa_pairs = []
    # 匹配 [用户] 开头的内容
    pattern = r'\[用户\]\s*(.+?)(?=\n\[AI\]|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        question = match.strip()
        if len(question) > 3 and len(question) < 500:
            qa_pairs.append({"question": question})
    
    return qa_pairs

def get_answer(content: str) -> str:
    """获取 [AI] 回答"""
    pattern = r'\[AI\]\s*(.+?)(?=\n\[用户\]|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    for m in matches:
        if len(m.strip()) > 0:
            return m.strip()[:200]  # 限制长度
    return ""

def main():
    print("=" * 60)
    print("测试集生成器 - 从 data/2026/04/10")
    print("=" * 60)
    
    # 收集所有节点
    md_files = sorted(SOURCE_DIR.glob("*.md"))
    # 排除 .DS_Store 等非md文件
    md_files = [f for f in md_files if f.suffix == '.md']
    print(f"\n找到 {len(md_files)} 个节点文件")
    
    # 随机抽取30个
    random.seed(42)  # 固定随机种子，保证可重复
    if len(md_files) >= 30:
        sample_files = random.sample(md_files, 30)
    else:
        sample_files = md_files
        print(f"警告: 只有{len(md_files)}个节点，不足30个")
    
    print(f"随机抽取 {len(sample_files)} 个节点...")
    
    questions = []
    for idx, node_path in enumerate(sample_files):
        node_id = node_path.stem
        
        try:
            with open(node_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取问答对
            qa_pairs = extract_qa_pairs(content)
            if qa_pairs:
                qa = qa_pairs[0]  # 取第一个问答对
                question = qa['question']
                answer = get_answer(content)
                
                # 使用问题本身的前10个词作为答案关键词（用于benchmark评分）
                answer_keywords = ' '.join(question.split()[:10])
                
                questions.append({
                    "question": question,
                    "answer": answer or answer_keywords,
                    "answer_id": node_id,
                    "type": "memory_node",
                    "date": "2026-04-10"
                })
                
                if idx < 5:
                    print(f"  [{idx+1}] {node_id}")
                    print(f"      Q: {question[:60]}...")
            else:
                # 没有问答对，使用标题
                title_match = re.search(r'title:\s*(.+)', content)
                title = title_match.group(1).strip() if title_match else node_id
                questions.append({
                    "question": title,
                    "answer": node_id,
                    "answer_id": node_id,
                    "type": "memory_node",
                    "date": "2026-04-10"
                })
                if idx < 5:
                    print(f"  [{idx+1}] {node_id} (title only)")
                
        except Exception as e:
            print(f"  Error processing {node_id}: {e}")
    
    # 构建测试集
    testset = {
        "description": "从 data/2026/04/10 随机抽取30个节点生成的测试集",
        "created": "2026-04-11T04:45:00",
        "source_dir": str(SOURCE_DIR),
        "total_questions": len(questions),
        "questions": questions
    }
    
    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 已保存到: {OUTPUT_PATH}")
    print(f"✓ 总计 {len(questions)} 个测试问题")
    
    # 显示前3个样本
    print("\n样本预览:")
    for i, q in enumerate(questions[:3]):
        print(f"\n[{i+1}] {q['answer_id']}")
        print(f"  Q: {q['question'][:80]}...")
        print(f"  A: {q['answer'][:80]}...")

if __name__ == "__main__":
    main()
