#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入 cli_direct.jsonl 到 Memora 记忆系统

使用 MemorySystem.add_memory_from_messages() 统一处理 skill 检测和内容格式化

Usage:
    python3 import_cli_direct.py --input /Users/rama/.nanobot/workspace/sessions/cli_direct.jsonl \
                                 [--max-sessions 100] \
                                 [--dry-run] \
                                 [--skip-existing]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Memora 路径
MEMORA_PATH = Path("/Users/rama/.nanobot/workspace/Memora")
sys.path.insert(0, str(MEMORA_PATH))

from src.memory_system import MemorySystem


def load_conversations(input_file: Path, max_sessions: Optional[int] = None, date_filter: Optional[str] = None) -> List[List[Dict]]:
    """
    从 cli_direct.jsonl 加载对话会话
    
    将连续的用户+助手+tool 消息组合成一个会话
    """
    conversations = []
    current_session = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                msg = json.loads(line)
                role = msg.get('role', '')
                
                # user 消息开始新会话
                if role == 'user':
                    if current_session:
                        conversations.append(current_session)
                        if max_sessions and len(conversations) >= max_sessions:
                            break
                    current_session = [msg]
                
                # assistant/tool 消息继续当前会话
                elif role in ('assistant', 'tool') and current_session:
                    current_session.append(msg)
                    
            except json.JSONDecodeError:
                continue
    
    # 添加最后一个会话
    if current_session and not (max_sessions and len(conversations) >= max_sessions):
        conversations.append(current_session)
    
    # 按日期过滤
    if date_filter:
        filtered = []
        for session in conversations:
            # 从第一条消息时间判断
            first_msg = session[0] if session else {}
            timestamp = first_msg.get('timestamp', '')
            if date_filter in str(timestamp):
                filtered.append(session)
        conversations = filtered
    
    return conversations


def main():
    parser = argparse.ArgumentParser(description='导入 cli_direct.jsonl 到 Memora')
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='输入的 cli_direct.jsonl 文件路径')
    parser.add_argument('--max-sessions', '-n', type=int, default=None,
                        help='最多导入多少组会话')
    parser.add_argument('--date', '-d', type=str, default=None,
                        help='只导入指定日期的会话 (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式，不实际保存')
    parser.add_argument('--skip-existing', '-s', action='store_true',
                        help='跳过已存在的内容（基于内容哈希）')
    parser.add_argument('--batch-size', '-b', type=int, default=50,
                        help='每多少条触发一次图构建（默认50）')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ 文件不存在: {args.input}")
        return 1
    
    print(f"📂 加载对话: {args.input}")
    
    # 加载会话
    conversations = load_conversations(
        args.input,
        max_sessions=args.max_sessions,
        date_filter=args.date
    )
    
    print(f"✅ 找到 {len(conversations)} 组对话")
    
    if args.dry_run:
        print("\n🔍 预览模式（不保存）:")
        for i, session in enumerate(conversations[:5], 1):
            user_msg = next((m for m in session if m.get('role') == 'user'), None)
            preview = user_msg.get('content', '')[:50] + "..." if user_msg else "N/A"
            print(f"  {i}. {preview}")
        if len(conversations) > 5:
            print(f"  ... 还有 {len(conversations) - 5} 组")
        return 0
    
    # 初始化 MemorySystem
    print("\n🧠 初始化 MemorySystem...")
    ms = MemorySystem()
    
    # 用于去重
    existing_hashes = set()
    if args.skip_existing:
        for node in ms.storage.get_all():
            content_hash = hash(node.content) % (2**32)
            existing_hashes.add(content_hash)
        print(f"   已加载 {len(existing_hashes)} 条现有记忆用于去重")
    
    # 导入统计
    imported = 0
    skipped = 0
    pending_nodes = []
    
    print(f"\n📝 开始导入...")
    
    for i, session in enumerate(conversations, 1):
        # 去重检查
        if args.skip_existing:
            # 简单去重：用用户第一条消息的内容哈希
            user_msg = next((m for m in session if m.get('role') == 'user'), None)
            if user_msg:
                content_hash = hash(user_msg.get('content', '')) % (2**32)
                if content_hash in existing_hashes:
                    skipped += 1
                    continue
        
        try:
            # 使用 MemorySystem 统一处理
            node = ms.add_memory_from_messages(
                messages=session,
                source="cli-import",
                base_tags=["cli-import"]
            )
            
            if node:
                imported += 1
                pending_nodes.append(node.url)
                
                # 每 batch_size 条触发一次图构建
                if args.batch_size > 0 and len(pending_nodes) >= args.batch_size:
                    print(f"\n  🔄 触发批量图构建（{len(pending_nodes)} 节点）...")
                    ms.build_graph(auto_link=True)
                    pending_nodes = []
            
            # 每10条显示进度
            if i % 10 == 0:
                print(f"   进度: {i}/{len(conversations)} (已导入 {imported})")
                
        except Exception as e:
            print(f"   ❌ 导入失败: {e}")
            continue
    
    # 最后统一构建图
    if pending_nodes:
        print(f"\n🔄 最终图构建（{len(pending_nodes)} 节点）...")
        ms.build_graph(auto_link=True)
    
    # 统计
    print(f"\n✅ 导入完成!")
    print(f"   成功: {imported}")
    print(f"   跳过: {skipped}")
    
    stats = ms.stats()
    print(f"\n📊 系统统计:")
    print(f"   总节点数: {stats['total_nodes']}")
    print(f"   总标签数: {stats['total_tags']}")
    print(f"   平均 PageRank: {stats['avg_pagerank']:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
