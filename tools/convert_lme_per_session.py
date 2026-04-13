#!/usr/bin/env python3
"""
Convert LongMemEval: 每个Session一个节点
测试版本：先处理第一个group的前3个sessions验证思路

Usage:
    python3 convert_lme_per_session.py --input ../benchmark/longmemeval_s_cleaned.json --output-dir ../data/test-per-session
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_date(date_str: str) -> datetime:
    """Parse: 2023/05/20 (Sat) 02:21"""
    try:
        cleaned = date_str.split("(")[0].strip() + " " + date_str.split(")")[-1].strip()
        return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
    except:
        return datetime.now()


def session_to_content(session: List[Dict], session_num: int, date: str) -> str:
    """单个session转markdown"""
    parts = [f"### Session {session_num} ({date})\n"]
    for msg in session:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"**User:** {content}")
        elif role == "assistant":
            parts.append(f"**Assistant:** {content}")
        else:
            parts.append(f"**{role.capitalize()}:** {content}")
    return "\n\n".join(parts)


def extract_simple_tags(content: str) -> List[str]:
    """简单关键词提取"""
    tags = ["longmemeval", "session"]
    # 简单规则提取
    keywords = ["学位", "毕业", "工作", "城市", "喜欢", "爱好", "姓名", "年龄"]
    for kw in keywords:
        if kw in content:
            tags.append(kw)
    return tags


def convert_one_session(
    session: List[Dict],
    session_id: str,
    date_str: str,
    session_num: int,
    group_id: str,
    output_dir: Path
) -> Dict:
    """转换单个session为节点"""
    
    date = parse_date(date_str)
    date_path = date.strftime("%Y/%m/%d")
    
    # 节点ID: lme-{group_id}-{session_num:04d}
    node_id = f"lme-{group_id}-{session_num:04d}"
    
    node_dir = output_dir / date_path
    node_dir.mkdir(parents=True, exist_ok=True)
    
    # 内容
    content = session_to_content(session, session_num, date_str)
    tags = extract_simple_tags(content)
    
    # 添加session特定标签
    tags.append(f"group:{group_id}")
    tags.append(f"session-num:{session_num}")
    
    # 生成markdown
    md_content = f"""---
id: {node_id}
timestamp: {date.isoformat()}
title: LongMemEval Session {session_num} (Group {group_id})
tags: {tags}
source: longmemeval
session_id: {session_id}
group_id: {group_id}
session_num: {session_num}
pagerank: 1.0
---

{content}
"""
    
    filepath = node_dir / f"{node_id}.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return {
        "id": node_id,
        "filepath": str(filepath),
        "group_id": group_id,
        "session_num": session_num,
        "date": date.isoformat()
    }


def convert_test(
    input_file: Path,
    output_dir: Path,
    max_groups: int = 1,
    max_sessions_per_group: int = 3
):
    """测试转换：只处理前N个group的前M个sessions"""
    
    print(f"Loading from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total groups: {len(data)}")
    print(f"Test config: {max_groups} groups × {max_sessions_per_group} sessions = {max_groups * max_sessions_per_group} nodes\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_created = []
    
    for entry in data[:max_groups]:
        group_id = entry["question_id"]
        sessions = entry.get("haystack_sessions", [])
        session_ids = entry.get("haystack_session_ids", [])
        dates = entry.get("haystack_dates", [])
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        
        print(f"Group: {group_id}")
        print(f"  Question: {question[:60]}...")
        print(f"  Answer: {answer}")
        print(f"  Total sessions: {len(sessions)}")
        print(f"  Creating nodes for sessions 1-{min(max_sessions_per_group, len(sessions))}...")
        
        for i in range(min(max_sessions_per_group, len(sessions))):
            node_info = convert_one_session(
                session=sessions[i],
                session_id=session_ids[i] if i < len(session_ids) else "",
                date_str=dates[i] if i < len(dates) else "",
                session_num=i + 1,
                group_id=group_id,
                output_dir=output_dir
            )
            nodes_created.append(node_info)
            print(f"    ✅ Session {i+1}: {node_info['id']}")
        
        print()
    
    print(f"{'='*60}")
    print(f"Test conversion complete!")
    print(f"  Nodes created: {len(nodes_created)}")
    print(f"  Output: {output_dir}")
    
    # 列出创建的节点
    print(f"\nCreated nodes:")
    for n in nodes_created:
        print(f"  - {n['id']}: {n['filepath']}")
    
    return nodes_created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=3)
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input not found: {args.input}")
        sys.exit(1)
    
    convert_test(
        input_file=args.input,
        output_dir=args.output_dir,
        max_groups=args.max_groups,
        max_sessions_per_group=args.max_sessions
    )


if __name__ == "__main__":
    main()
