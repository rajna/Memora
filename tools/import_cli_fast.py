#!/usr/bin/env python3
"""
快速导入 CLI 会话历史到 Memora - 使用简化嵌入后端
用法: python import_cli_fast.py <jsonl_file> <start_date> <end_date>
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')

def parse_args():
    if len(sys.argv) < 4:
        print("用法: python import_cli_fast.py <jsonl_file> <start_date> <end_date>")
        print("示例: python import_cli_fast.py sessions/cli_direct.jsonl 2026-04-01 2026-04-12")
        sys.exit(1)
    return sys.argv[1], sys.argv[2], sys.argv[3]


def load_and_group(input_file, start_date, end_date):
    """加载并分组消息"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"1. 正在加载 {start_date} 到 {end_date} 的消息...")
    
    conversations = []
    current_session = []
    msg_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get('_type') == 'metadata':
                    continue
                
                timestamp_str = msg.get('timestamp', '')
                if not timestamp_str:
                    continue
                
                try:
                    msg_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if not (start_dt.date() <= msg_dt.date() <= end_dt.date()):
                        continue
                except ValueError:
                    continue
                
                msg_count += 1
                role = msg.get('role')
                
                if role == 'user':
                    if current_session:
                        conversations.append(current_session)
                    current_session = [msg]
                elif role in ('assistant', 'tool') and current_session:
                    current_session.append(msg)
                    
            except json.JSONDecodeError:
                continue
    
    if current_session:
        conversations.append(current_session)
    
    print(f"   找到 {msg_count} 条消息, {len(conversations)} 组对话")
    return conversations


def main():
    input_file, start_date, end_date = parse_args()
    
    if not Path(input_file).exists():
        print(f"错误: 文件不存在")
        sys.exit(1)
    
    # 加载对话
    conversations = load_and_group(input_file, start_date, end_date)
    if not conversations:
        print("没有数据需要导入")
        return
    
    print(f"\n2. 准备导入 {len(conversations)} 组对话")
    
    # 初始化 MemorySystem - 使用 tfidf 后端（更快）
    print("3. 初始化 MemorySystem (使用TF-IDF后端)...")
    from src.memory_system import MemorySystem
    ms = MemorySystem(
        memory_dir="/Users/rama/.nanobot/workspace/Memora/data",
        embedding_backend="tfidf"  # 使用更快的TF-IDF
    )
    
    # 导入
    print(f"4. 开始导入...")
    imported = 0
    failed = 0
    skipped = 0
    
    for i, session in enumerate(conversations, 1):
        if i % 50 == 0:
            print(f"   进度: {i}/{len(conversations)} (成功:{imported} 失败:{failed} 跳过:{skipped})")
        
        try:
            node = ms.add_memory_from_messages(
                messages=session,
                source="cli-import",
                base_tags=["cli-import", start_date[:7]]
            )
            if node:
                imported += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            if failed <= 5:  # 只显示前5个错误
                print(f"   错误 [{i}]: {str(e)[:80]}")
    
    # 保存
    print("\n5. 保存索引...")
    try:
        ms.save()
        print("   ✅ 已保存")
    except Exception as e:
        print(f"   警告: {e}")
    
    # 统计
    print(f"\n✅ 导入完成!")
    print(f"   成功: {imported}")
    print(f"   失败: {failed}")
    print(f"   跳过: {skipped}")
    print(f"   总计: {len(conversations)}")


if __name__ == "__main__":
    main()
