#!/usr/bin/env python3
"""
Convert LongMemEval dataset to Memora node format
将 LongMemEval 数据转换为记忆系统节点格式

Usage:
  python3 tools/convert_longmemeval.py --input benchmark/longmemeval_s_cleaned.json --output data/2026/04/10test
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.models import MemoryNode
except ImportError:
    # Fallback if models not available
    MemoryNode = None


def parse_date(date_str: str) -> datetime:
    """Parse LongMemEval date format: 2023/05/30 (Tue) 23:40"""
    try:
        # Remove day of week: "2023/05/30 (Tue) 23:40" -> "2023/05/30 23:40"
        parts = date_str.split(' (')
        date_part = parts[0]
        time_part = parts[1].split(') ')[1] if ') ' in parts[1] else parts[1].replace(')', '')
        return datetime.strptime(f"{date_part} {time_part}", "%Y/%m/%d %H:%M")
    except:
        return datetime.now()


def session_to_content(session: list) -> str:
    """Convert a session (list of messages) to text content"""
    lines = []
    for msg in session:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        lines.append(f"[{role.upper()}]: {content}")
    return "\n\n".join(lines)


def generate_node_id(timestamp: datetime, content_hash: str) -> str:
    """Generate unique node ID"""
    ts_str = timestamp.strftime("%Y%m%d%H%M%S")
    short_hash = content_hash[:8]
    return f"{ts_str}_{short_hash}"


def create_node_from_session(
    session: list,
    session_idx: int,
    question_id: str,
    session_date: str,
    output_dir: Path
) -> dict:
    """Create a memory node from a LongMemEval session"""
    
    content = session_to_content(session)
    content_hash = hashlib.md5(content.encode()).hexdigest()
    
    # Parse timestamp
    timestamp = parse_date(session_date)
    
    # Generate node ID
    node_id = generate_node_id(timestamp, content_hash)
    
    # Determine node type based on content
    if any("user" in msg.get("role", "") for msg in session):
        node_type = "conversation"
    else:
        node_type = "note"
    
    # Create node data
    node = {
        "id": node_id,
        "url": f"/memory/test/{question_id}/{session_idx}",
        "title": f"Session {session_idx} - {question_id[:8]}",
        "content": content,
        "node_type": node_type,
        "tags": ["longmemeval", f"q_{question_id[:8]}", "test_data"],
        "created": timestamp.isoformat(),
        "modified": timestamp.isoformat(),
        "source": "LongMemEval",
        "source_question_id": question_id,
        "metadata": {
            "session_date": session_date,
            "session_idx": session_idx,
            "messages_count": len(session)
        }
    }
    
    return node


def convert_question(
    question_data: dict,
    output_dir: Path
) -> tuple[int, list[str]]:
    """Convert one question's haystack sessions to nodes"""
    
    question_id = question_data['question_id']
    haystack_sessions = question_data['haystack_sessions']
    haystack_dates = question_data['haystack_dates']
    
    node_ids = []
    
    for idx, (session, session_date) in enumerate(zip(haystack_sessions, haystack_dates)):
        node = create_node_from_session(
            session=session,
            session_idx=idx,
            question_id=question_id,
            session_date=session_date,
            output_dir=output_dir
        )
        
        # Save node to file
        node_file = output_dir / f"{node['id']}.json"
        with open(node_file, 'w', encoding='utf-8') as f:
            json.dump(node, f, ensure_ascii=False, indent=2)
        
        node_ids.append(node['id'])
    
    return len(haystack_sessions), node_ids


def main():
    parser = argparse.ArgumentParser(description='Convert LongMemEval to memory nodes')
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions (for testing)')
    parser.add_argument('--questions', type=str, default=None, help='Comma-separated question IDs to process')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total questions in dataset: {len(data)}")
    
    # Filter questions if specified
    if args.questions:
        target_ids = set(args.questions.split(','))
        data = [q for q in data if q['question_id'] in target_ids]
        print(f"Processing {len(data)} specified questions")
    
    if args.limit:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} questions")
    
    # Convert each question
    total_nodes = 0
    question_node_map = {}
    
    for i, question_data in enumerate(data, 1):
        qid = question_data['question_id']
        question_text = question_data['question']
        
        print(f"\n[{i}/{len(data)}] Processing question: {qid}")
        print(f"    Q: {question_text[:80]}...")
        
        count, node_ids = convert_question(question_data, output_dir)
        question_node_map[qid] = {
            'node_ids': node_ids,
            'question': question_data['question'],
            'answer': question_data['answer'],
            'question_date': question_data['question_date'],
            'question_type': question_data['question_type']
        }
        
        print(f"    -> {count} nodes created")
        total_nodes += count
    
    # Save question-to-node mapping
    mapping_file = output_dir / '_question_node_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(question_node_map, f, ensure_ascii=False, indent=2)
    
    # Save summary
    summary = {
        'total_questions': len(data),
        'total_nodes': total_nodes,
        'avg_nodes_per_question': total_nodes / len(data) if data else 0,
        'output_directory': str(output_dir),
        'mapping_file': str(mapping_file)
    }
    
    summary_file = output_dir / '_conversion_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Total questions: {summary['total_questions']}")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Avg nodes/question: {summary['avg_nodes_per_question']:.1f}")
    print(f"  Output: {output_dir}")
    print(f"  Mapping: {mapping_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
