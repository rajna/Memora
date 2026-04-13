#!/usr/bin/env python3
"""
批量导入 CLI 会话历史到 Memora，支持分批处理和断点续传
用法: python import_cli_batch.py <jsonl_file> <start_date> <end_date> [batch_size] [offset]
示例: python import_cli_batch.py sessions/cli_direct.jsonl 2026-04-01 2026-04-12 100 0
"""

import json
import sys
import pickle
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem

PROGRESS_FILE = Path("/Users/rama/.nanobot/workspace/Memora/.import_progress.pkl")

def parse_args():
    if len(sys.argv) < 4:
        print("用法: python import_cli_batch.py <jsonl_file> <start_date> <end_date> [batch_size] [offset]")
        print("日期格式: YYYY-MM-DD")
        print("示例: python import_cli_batch.py sessions/cli_direct.jsonl 2026-04-01 2026-04-12 100 0")
        sys.exit(1)
    
    input_file = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    
    return input_file, start_date, end_date, batch_size, offset


def load_messages_by_date(input_file, start_date, end_date):
    """加载指定日期范围内的消息"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    messages = []
    print(f"1. 正在加载 {input_file} 中 {start_date} 到 {end_date} 的消息...")
    
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
                    if start_dt.date() <= msg_dt.date() <= end_dt.date():
                        messages.append(msg)
                except ValueError:
                    continue
                    
            except json.JSONDecodeError:
                continue
    
    print(f"   找到 {len(messages)} 条消息")
    return messages


def group_into_conversations(messages):
    """将消息按对话分组"""
    print("2. 正在将消息分组为对话...")
    
    conversations = []
    current_session = []
    
    for msg in messages:
        role = msg.get('role')
        
        if role == 'user':
            if current_session:
                conversations.append(current_session)
            current_session = [msg]
        elif role in ('assistant', 'tool') and current_session:
            current_session.append(msg)
    
    if current_session:
        conversations.append(current_session)
    
    print(f"   共 {len(conversations)} 组对话")
    return conversations


def save_progress(offset, imported_count, total_count):
    """保存导入进度"""
    progress = {
        'offset': offset,
        'imported': imported_count,
        'total': total_count,
        'timestamp': datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump(progress, f)


def load_progress():
    """加载导入进度"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def import_batch(ms, conversations, start_idx, batch_size, start_date):
    """导入一批对话"""
    end_idx = min(start_idx + batch_size, len(conversations))
    batch = conversations[start_idx:end_idx]
    
    imported = 0
    failed = 0
    
    for i, session in enumerate(batch, start_idx + 1):
        first_msg = session[0] if session else None
        if not first_msg:
            continue
        
        user_content = first_msg.get('content', '')[:40]
        
        try:
            node = ms.add_memory_from_messages(
                messages=session,
                source="cli-import",
                base_tags=["cli-import", start_date[:7]]
            )
            if node:
                imported += 1
                print(f"  [{i}/{len(conversations)}] ✅ {user_content}...")
            else:
                print(f"  [{i}/{len(conversations)}] ⚠️ 跳过 (空内容)")
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(conversations)}] ❌ 失败: {str(e)[:50]}")
    
    return imported, failed


def main():
    input_file, start_date, end_date, batch_size, offset = parse_args()
    
    if not Path(input_file).exists():
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    # 检查是否有之前的进度
    progress = load_progress()
    if progress and offset == 0:
        print(f"\n发现之前的导入进度: {progress['imported']}/{progress['total']} 已导入")
        response = input("是否继续上次的进度? (y/n/s for start over): ")
        if response.lower() == 'y':
            offset = progress['offset']
        elif response.lower() == 's':
            PROGRESS_FILE.unlink(missing_ok=True)
    
    # 加载消息
    messages = load_messages_by_date(input_file, start_date, end_date)
    if not messages:
        print(f"未找到 {start_date} 到 {end_date} 之间的消息")
        sys.exit(0)
    
    # 分组
    conversations = group_into_conversations(messages)
    if not conversations:
        print("没有有效的对话可以导入")
        sys.exit(0)
    
    total = len(conversations)
    
    if offset >= total:
        print(f"所有 {total} 组对话已导入完成!")
        PROGRESS_FILE.unlink(missing_ok=True)
        sys.exit(0)
    
    print(f"\n3. 准备从第 {offset+1} 组开始导入 (共 {total} 组, 批次大小 {batch_size})")
    
    # 初始化 MemorySystem
    print("4. 正在初始化 MemorySystem...")
    ms = MemorySystem(memory_dir="/Users/rama/.nanobot/workspace/Memora/data")
    
    # 分批导入
    total_imported = progress['imported'] if progress else 0
    
    while offset < total:
        print(f"\n--- 批次: {offset+1}-{min(offset+batch_size, total)} ---")
        imported, failed = import_batch(ms, conversations, offset, batch_size, start_date)
        
        total_imported += imported
        offset += batch_size
        
        # 保存进度
        save_progress(offset, total_imported, total)
        
        print(f"本批次: 成功 {imported}, 失败 {failed}")
        print(f"总计: {total_imported}/{total} 已导入")
        
        # 每批保存一次
        try:
            ms.save()
            print("索引已保存")
        except Exception as e:
            print(f"保存警告: {e}")
    
    # 完成
    PROGRESS_FILE.unlink(missing_ok=True)
    print(f"\n✅ 导入完成!")
    print(f"   总计导入: {total_imported}/{total} 组对话")
    print(f"   日期范围: {start_date} 到 {end_date}")


if __name__ == "__main__":
    main()
