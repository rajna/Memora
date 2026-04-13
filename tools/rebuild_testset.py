#!/usr/bin/env python3
"""
重新构建 LongMemEval 80题测试集
- 50个 LongMemEval 问题（从 lme-*.md 节点提取）
- 30个 Memory System 问题（从今天14:00前的节点提取）
"""

import json
import frontmatter
import re
from pathlib import Path
from datetime import datetime

def extract_lme_questions():
    """从 longmemeval_s_cleaned.json 读取正确的测试问题"""
    benchmark_dir = Path(__file__).parent.parent / "benchmark"
    data_dir = Path(__file__).parent.parent / "data"
    
    # 加载 LongMemEval 原始数据
    lme_json_path = benchmark_dir / "longmemeval_s_cleaned.json"
    if not lme_json_path.exists():
        print(f"   ⚠️  未找到 {lme_json_path}")
        return []
    
    with open(lme_json_path, 'r', encoding='utf-8') as f:
        lme_data = json.load(f)
    
    questions = []
    skipped = 0
    for item in lme_data:
        qid = item.get('question_id')
        question = str(item.get('question', '')).strip()
        answer = str(item.get('answer', '')).strip()
        
        if not qid or not question:
            continue
        
        # 查找对应的节点文件 lme-{qid}-*.md
        node_files = list(data_dir.rglob(f"lme-{qid}-*.md"))
        if not node_files:
            skipped += 1
            continue  # 静默跳过，不打印警告
        
        node_id = node_files[0].stem  # e.g., lme-58ef2f1c-0109
        
        questions.append({
            "question": question,
            "answer": answer,
            "answer_id": node_id,
            "type": "longmemeval",
            "question_id": qid
        })
    
    if skipped > 0:
        print(f"   (跳过 {skipped} 个缺失节点)")
    
    return questions

def extract_memory_questions():
    """从今天14:00前的 memory system 节点提取问题"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # 找到今天14:00前的节点
    cutoff = datetime(2026, 4, 10, 14, 0, 0)
    
    questions = []
    memory_files = list((data_dir / "2026" / "04" / "10").glob("*.md")) if (data_dir / "2026" / "04" / "10").exists() else []
    
    for f in sorted(memory_files):
        post = frontmatter.load(f)
        created = post.get('created')
        
        if not created:
            continue
            
        # 解析时间
        try:
            dt = datetime.fromisoformat(created.replace('Z', '+00:00').replace('+00:00', ''))
            if dt >= cutoff:
                continue
        except:
            continue
        
        # 从内容中提取用户问题（查找 ### 用户 或 **用户** 或 [用户]）
        content = post.content
        
        # 尝试多种格式
        patterns = [
            r'(?:###\s*\[?用户\]?|\*\*用户\*\*|\[用户\])[:：]?\s*(.+?)(?=\n\n|\n(?:###|\*\*AI|> |$))',
            r'^(?!#)(.+?)(?=\n\n|###|\*\*AI)',  # 第一行非标题内容
        ]
        
        question = None
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                question = match.group(1).strip().replace('\n', ' ')
                if len(question) > 10:  # 至少10个字符
                    break
        
        if question and len(question) > 10:
            # 截断
            if len(question) > 200:
                question = question[:200] + "..."
            
            node_id = f.stem  # e.g., 202604101004-ce7aa689
            
            # 尝试分类
            category = "general"
            title = post.get('title', '') or ''
            tags = post.get('tags', [])
            
            # 基于标题和标签分类
            if any(k in title.lower() + ' '.join(tags).lower() for k in ['benchmark', 'test', 'r@']):
                category = "benchmark"
            elif any(k in title.lower() + ' '.join(tags).lower() for k in ['pagerank', 'pr']):
                category = "pagerank"
            elif any(k in title.lower() + ' '.join(tags).lower() for k in ['embedding', 'vector']):
                category = "embedding"
            elif any(k in title.lower() + ' '.join(tags).lower() for k in ['retrieval', 'search']):
                category = "retrieval"
            
            questions.append({
                "question": question,
                "answer_id": node_id,
                "type": "memory_system",
                "category": category
            })
    
    return questions

def main():
    print("=" * 70)
    print("重新构建 LongMemEval 测试集")
    print("=" * 70)
    
    # 提取 LongMemEval 问题
    print("\n📌 提取 LongMemEval 问题...")
    lme_questions = extract_lme_questions()
    print(f"   找到 {len(lme_questions)} 个 LME 问题")
    
    # 如果超过50个，取前50
    lme_questions = lme_questions[:50]
    print(f"   使用 {len(lme_questions)} 个")
    
    # 提取 Memory System 问题
    print("\n📌 提取 Memory System 问题 (今天14:00前)...")
    memory_questions = extract_memory_questions()
    print(f"   找到 {len(memory_questions)} 个问题")
    
    # 如果超过30个，按时间排序取前30
    memory_questions = memory_questions[:30]
    print(f"   使用 {len(memory_questions)} 个")
    
    # 合并
    all_questions = lme_questions + memory_questions
    
    # 构建测试集
    testset = {
        "description": "LongMemEval 50-Test + Memory System 30-Test (14:00前节点) 合并测试集",
        "created": datetime.now().isoformat(),
        "total_questions": len(all_questions),
        "source": {
            "longmemeval": len(lme_questions),
            "memory_system": len(memory_questions)
        },
        "questions": all_questions
    }
    
    # 保存
    output_path = Path(__file__).parent.parent / "benchmark" / "longmemeval_test_80.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 测试集已保存: {output_path}")
    print(f"   - LongMemEval: {len(lme_questions)} 题")
    print(f"   - Memory System: {len(memory_questions)} 题")
    print(f"   - 总计: {len(all_questions)} 题")
    
    # 显示样例
    print("\n📋 样例问题:")
    print("\n[LME 示例]")
    print(f"  Q: {lme_questions[0]['question'][:100]}...")
    print(f"  A: {lme_questions[0]['answer_id']}")
    
    if memory_questions:
        print("\n[Memory System 示例]")
        print(f"  Q: {memory_questions[0]['question'][:100]}...")
        print(f"  A: {memory_questions[0]['answer_id']}")
        print(f"  Category: {memory_questions[0]['category']}")

if __name__ == "__main__":
    main()
