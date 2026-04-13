#!/usr/bin/env python3
"""
Convert LongMemEval: 每个Session一个节点 (完整版)
50 groups × ~50 sessions = ~2500 nodes

Usage:
    python3 convert_lme_all_sessions.py \
        --input ../benchmark/longmemeval_s_cleaned.json \
        --output-dir ../data/lme-per-session
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


def extract_tags(content: str) -> List[str]:
    """提取关键词标签"""
    tags = ["longmemeval", "session"]
    keywords = ["学位", "毕业", "工作", "城市", "喜欢", "爱好", "姓名", "年龄",
                "degree", "graduated", "work", "job", "city", "hobby", "name"]
    content_lower = content.lower()
    for kw in keywords:
        if kw in content_lower:
            tags.append(kw)
    return list(set(tags))


def convert_session(
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
    
    node_id = f"lme-{group_id}-{session_num:04d}"
    
    node_dir = output_dir / date_path
    node_dir.mkdir(parents=True, exist_ok=True)
    
    content = session_to_content(session, session_num, date_str)
    tags = extract_tags(content)
    tags.extend([f"group:{group_id}", f"session-num:{session_num}"])
    
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


def convert_all(
    input_file: Path,
    output_dir: Path,
    max_groups: int = None
):
    """转换所有sessions"""
    
    print(f"Loading from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_groups = len(data)
    if max_groups:
        data = data[:max_groups]
        print(f"Processing {max_groups}/{total_groups} groups...")
    else:
        print(f"Processing all {total_groups} groups...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nodes_created = []
    total_sessions = 0
    
    for entry in data:
        group_id = entry["question_id"]
        sessions = entry.get("haystack_sessions", [])
        session_ids = entry.get("haystack_session_ids", [])
        dates = entry.get("haystack_dates", [])
        
        if not sessions:
            continue
        
        total_sessions += len(sessions)
        
        for i in range(len(sessions)):
            node_info = convert_session(
                session=sessions[i],
                session_id=session_ids[i] if i < len(session_ids) else "",
                date_str=dates[i] if i < len(dates) else "",
                session_num=i + 1,
                group_id=group_id,
                output_dir=output_dir
            )
            nodes_created.append(node_info)
        
        print(f"  ✅ Group {group_id}: {len(sessions)} sessions -> nodes")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Groups processed: {len(data)}")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Nodes created: {len(nodes_created)}")
    print(f"  Output: {output_dir}")
    
    # Save manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source": str(input_file),
        "total_groups": total_groups,
        "processed_groups": len(data),
        "total_sessions": total_sessions,
        "nodes_created": len(nodes_created),
        "output_dir": str(output_dir)
    }
    
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_file}")
    
    return nodes_created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--max-groups", "-n", type=int, default=None,
                        help="Limit number of groups (for testing)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input not found: {args.input}")
        sys.exit(1)
    
    convert_all(
        input_file=args.input,
        output_dir=args.output_dir,
        max_groups=args.max_groups
    )


if __name__ == "__main__":
    main()
