#!/usr/bin/env python3
"""
从 2026 年数据中均匀采样 50 个节点生成测试集
- 尽量分布均匀（按日期分层采样）
- 生成格式与 test_2026_04_10_30.json 一致
"""

import json
import frontmatter
import re
import random
from pathlib import Path
from datetime import datetime

def collect_all_nodes():
    """收集 2026 年所有节点"""
    data_dir = Path(__file__).parent.parent / "data" / "2026"
    
    nodes_by_date = {}
    
    for month_dir in sorted(data_dir.glob("*")):
        if not month_dir.is_dir():
            continue
        month = month_dir.name
        
        for day_dir in sorted(month_dir.glob("*")):
            if not day_dir.is_dir():
                continue
            day = day_dir.name
            date_key = f"2026-{month}-{day}"
            
            nodes = []
            for md_file in sorted(day_dir.glob("*.md")):
                try:
                    post = frontmatter.load(md_file)
                    nodes.append({
                        "path": md_file,
                        "id": md_file.stem,
                        "date": date_key,
                        "title": post.get("title", ""),
                        "content": post.content,
                        "tags": post.get("tags", []),
                        "created": post.get("created", "")
                    })
                except Exception as e:
                    print(f"   ⚠️  跳过 {md_file}: {e}")
                    continue
            
            if nodes:
                nodes_by_date[date_key] = nodes
    
    return nodes_by_date

def extract_question_from_content(content, title=""):
    """从节点内容中提取问题（作为查询）"""
    # 清理内容
    content = content.strip()
    
    # 尝试多种格式提取
    patterns = [
        # 匹配 ### 用户 或 **用户** 格式
        r'(?:###\s*\[?用户\]?|\*\*用户\*\*|\[用户\])[:：]?\s*(.+?)(?=\n\n|\n(?:###|\*\*AI|> |$))',
        # 匹配 # 开头的标题后的第一行内容
        r'^#+\s*.+\n+(.+?)(?=\n\n|#|$)',
        # 匹配普通段落（第一行非空内容）
        r'^(?!#|\s*$|\[)(.+?)(?=\n\n|#|\[|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if match:
            question = match.group(1).strip()
            # 清理
            question = re.sub(r'\[AI\]|\[用户\]|\*\*|###|#', '', question)
            question = question.replace('\n', ' ').strip()
            
            if len(question) > 15:  # 至少15个字符
                return question[:300]  # 限制长度
    
    # 如果提取失败，使用标题或内容前100字符
    if title and len(title) > 5:
        return title[:200]
    
    # 使用内容前100字符
    clean_content = re.sub(r'#|\*|\[|\]|\(|\)|`', '', content[:200]).strip()
    if len(clean_content) > 15:
        return clean_content[:200]
    
    return None

def uniform_sample(nodes_by_date, total_samples=50):
    """均匀采样：按日期分层，每层采样尽量均匀"""
    dates = sorted(nodes_by_date.keys())
    total_dates = len(dates)
    
    print(f"\n📅 共有 {total_dates} 个日期，需要采样 {total_samples} 个节点")
    
    # 计算每个日期应该采样的数量
    base_count = total_samples // total_dates
    remainder = total_samples % total_dates
    
    sampled = []
    
    for i, date in enumerate(dates):
        nodes = nodes_by_date[date]
        # 前 remainder 个日期多采一个
        target = base_count + (1 if i < remainder else 0)
        
        if len(nodes) <= target:
            # 如果节点数不足，全部采样
            selected = nodes
        else:
            # 随机采样
            selected = random.sample(nodes, target)
        
        for node in selected:
            question = extract_question_from_content(node["content"], node["title"])
            if question:
                sampled.append({
                    "question": question,
                    "answer": node["content"][:500] + "..." if len(node["content"]) > 500 else node["content"],
                    "answer_id": node["id"],
                    "type": "memory_node",
                    "date": date
                })
        
        print(f"   {date}: {len(nodes):4d} 节点 → 采样 {len([s for s in sampled if s['date'] == date])} 个")
    
    return sampled

def main():
    random.seed(42)  # 可重复
    
    print("=" * 70)
    print("🎲 生成均匀分布的 2026 年测试集 (50 题)")
    print("=" * 70)
    
    # 收集所有节点
    print("\n📌 收集 2026 年所有节点...")
    nodes_by_date = collect_all_nodes()
    total_nodes = sum(len(nodes) for nodes in nodes_by_date.values())
    print(f"   找到 {total_nodes} 个节点，分布在 {len(nodes_by_date)} 个日期")
    
    # 均匀采样
    print("\n📌 均匀采样...")
    questions = uniform_sample(nodes_by_date, total_samples=50)
    
    # 如果采样不足50个，从剩余节点中补充
    if len(questions) < 50:
        print(f"\n⚠️  只提取到 {len(questions)} 个有效问题，尝试补充...")
        # 收集已使用的节点 ID
        used_ids = {q["answer_id"] for q in questions}
        
        # 从剩余节点中补充
        remaining = []
        for date, nodes in nodes_by_date.items():
            for node in nodes:
                if node["id"] not in used_ids:
                    question = extract_question_from_content(node["content"], node["title"])
                    if question:
                        remaining.append({
                            "question": question,
                            "answer": node["content"][:500] + "..." if len(node["content"]) > 500 else node["content"],
                            "answer_id": node["id"],
                            "type": "memory_node",
                            "date": date
                        })
        
        # 随机补充
        needed = 50 - len(questions)
        if len(remaining) >= needed:
            questions.extend(random.sample(remaining, needed))
        else:
            questions.extend(remaining)
    
    # 构建测试集
    testset = {
        "description": "从 data/2026 均匀采样50个节点生成的测试集",
        "created": datetime.now().isoformat(),
        "source_dir": "/Users/rama/.nanobot/workspace/Memora/data/2026",
        "total_questions": len(questions),
        "questions": questions
    }
    
    # 保存
    output_path = Path(__file__).parent.parent / "benchmark" / "test_2026_04_13_50.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 测试集已保存: {output_path}")
    print(f"   总计: {len(questions)} 题")
    
    # 显示分布统计
    date_dist = {}
    for q in questions:
        date_dist[q["date"]] = date_dist.get(q["date"], 0) + 1
    
    print("\n📊 日期分布:")
    for date in sorted(date_dist.keys()):
        print(f"   {date}: {date_dist[date]} 题")
    
    # 显示样例
    print("\n📋 样例问题:")
    for i, q in enumerate(questions[:3], 1):
        print(f"\n[{i}] {q['date']}")
        print(f"   Q: {q['question'][:100]}...")
        print(f"   A_ID: {q['answer_id']}")

if __name__ == "__main__":
    main()
