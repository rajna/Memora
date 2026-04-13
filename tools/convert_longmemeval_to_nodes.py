#!/usr/bin/env python3
"""
Convert LongMemEval haystack_sessions to Memora nodes
将 LongMemEval 对话数据转换为记忆节点

Usage:
    python3 convert_longmemeval_to_nodes.py --input ../benchmark/longmemeval_s_cleaned.json --output-dir ../data/2026/04/10test --max-questions 10
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_date(date_str: str) -> datetime:
    """Parse LongMemEval date format: 2023/05/20 (Sat) 02:21"""
    try:
        # Remove day of week: "2023/05/20 (Sat) 02:21" -> "2023/05/20 02:21"
        cleaned = date_str.split("(")[0].strip() + " " + date_str.split(")")[-1].strip()
        return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
    except:
        return datetime.now()


def session_to_node_content(session: List[Dict[str, Any]]) -> str:
    """Convert a session (list of messages) to markdown content"""
    parts = []
    for msg in session:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Format as readable conversation
        if role == "user":
            parts.append(f"**User:** {content}")
        elif role == "assistant":
            parts.append(f"**Assistant:** {content}")
        else:
            parts.append(f"**{role.capitalize()}:** {content}")
    
    return "\n\n".join(parts)


def convert_sessions_to_node(
    sessions: List[List[Dict]],
    session_ids: List[str],
    dates: List[str],
    question_id: str,
    output_dir: Path
) -> Dict[str, Any]:
    """Convert a list of sessions to a memory node"""
    
    # Use first session date as node date
    first_date = parse_date(dates[0]) if dates else datetime.now()
    date_path = first_date.strftime("%Y/%m/%d")
    
    # Create directory
    node_dir = output_dir / date_path
    node_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate node ID
    node_id = f"lme-{question_id}-{first_date.strftime('%H%M')}"
    
    # Build content
    content_parts = []
    content_parts.append(f"# LongMemEval Session Group\n")
    content_parts.append(f"**Question ID:** {question_id}\n")
    content_parts.append(f"**Sessions:** {len(sessions)}\n")
    content_parts.append(f"**Date Range:** {dates[0] if dates else 'unknown'} ~ {dates[-1] if dates else 'unknown'}\n")
    content_parts.append(f"---\n")
    
    # Add each session
    for i, (session, sid, date) in enumerate(zip(sessions, session_ids, dates)):
        content_parts.append(f"\n### Session {i+1} ({date})\n")
        content_parts.append(session_to_node_content(session))
    
    content = "\n".join(content_parts)
    
    # Generate tags from content (simple keyword extraction)
    tags = ["longmemeval", "test-data"]
    
    # Create markdown file
    md_content = f"""---
id: {node_id}
timestamp: {first_date.isoformat()}
title: LongMemEval {question_id} ({len(sessions)} sessions)
tags: {tags}
source: longmemeval
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
        "session_count": len(sessions),
        "date": first_date.isoformat()
    }


def convert_longmemeval(
    input_file: Path,
    output_dir: Path,
    max_questions: int = None
):
    """Main conversion function"""
    
    print(f"Loading LongMemEval data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_questions = len(data)
    print(f"Total questions: {total_questions}")
    
    if max_questions:
        data = data[:max_questions]
        print(f"Processing first {max_questions} questions...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each question
    nodes_created = []
    for entry in data:
        question_id = entry["question_id"]
        sessions = entry.get("haystack_sessions", [])
        session_ids = entry.get("haystack_session_ids", [])
        dates = entry.get("haystack_dates", [])
        
        if not sessions:
            print(f"  ⚠️  {question_id}: No sessions, skipping")
            continue
        
        node_info = convert_sessions_to_node(
            sessions=sessions,
            session_ids=session_ids,
            dates=dates,
            question_id=question_id,
            output_dir=output_dir
        )
        nodes_created.append(node_info)
        print(f"  ✅ {question_id}: {node_info['session_count']} sessions -> {node_info['filepath']}")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Nodes created: {len(nodes_created)}")
    print(f"  Output directory: {output_dir}")
    
    # Save manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source": str(input_file),
        "total_questions": total_questions,
        "processed_questions": len(data),
        "nodes_created": len(nodes_created),
        "nodes": nodes_created
    }
    
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_file}")
    
    return nodes_created


def main():
    parser = argparse.ArgumentParser(description="Convert LongMemEval to memory nodes")
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input longmemeval_s_cleaned.json file")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Output directory for memory nodes")
    parser.add_argument("--max-questions", "-n", type=int, default=None,
                        help="Maximum number of questions to process (for testing)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    convert_longmemeval(
        input_file=args.input,
        output_dir=args.output_dir,
        max_questions=args.max_questions
    )


if __name__ == "__main__":
    main()
