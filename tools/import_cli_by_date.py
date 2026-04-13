#!/usr/bin/env python3
"""
导入 CLI 会话历史到 Memora，支持按日期范围筛选
用法: python import_cli_by_date.py <jsonl_file> <start_date> <end_date>
示例: python import_cli_by_date.py sessions/cli_direct.jsonl 2026-04-01 2026-04-12
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# 添加 Memora 到路径
sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem


def parse_args():
    if len(sys.argv) < 4:
        print("用法: python import_cli_by_date.py <jsonl_file> <start_date> <end_date>")
        print("日期格式: YYYY-MM-DD")
        print("示例: python import_cli_by_date.py sessions/cli_direct.jsonl 2026-04-01 2026-04-12")
        sys.exit(1)
    
    input_file = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    
    return input_file, start_date, end_date


def load_messages_by_date(input_file, start_date, end_date):
    """加载指定日期范围内的消息"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    messages = []
    print(f"1. 正在加载 {input_file} 中 {start_date} 到 {end_date} 的消息...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                # 跳过 metadata 行
                if msg.get('_type') == 'metadata':
                    continue
                
                # 解析时间戳
                timestamp_str = msg.get('timestamp', '')
                if not timestamp_str:
                    continue
                
                try:
                    msg_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # 检查是否在日期范围内
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
            # 如果当前有对话，先保存
            if current_session:
                conversations.append(current_session)
            current_session = [msg]
        elif role in ('assistant', 'tool') and current_session:
            current_session.append(msg)
        # 其他情况忽略
    
    # 保存最后一个对话
    if current_session:
        conversations.append(current_session)
    
    print(f"   共 {len(conversations)} 组对话")
    return conversations


def import_to_memora(conversations, start_date, end_date):
    """导入对话到 Memora"""
    print("3. 正在初始化 MemorySystem...")
    
    memory_dir = "/Users/rama/.nanobot/workspace/Memora/data"
    ms = MemorySystem(memory_dir=memory_dir)
    
    print(f"   Memory dir: {memory_dir}")
    print(f"   Embedding backend: {ms.embedding_manager.embedding_backend.__class__.__name__ if hasattr(ms, 'embedding_manager') else 'N/A'}")
    
    print(f"\n4. 开始导入 {len(conversations)} 组对话...")
    imported_count = 0
    
    for i, session in enumerate(conversations, 1):
        # 获取第一消息的时间戳
        first_msg = session[0] if session else None
        if not first_msg:
            continue
        
        timestamp = first_msg.get('timestamp', '')
        
        # 创建标题
        user_content = first_msg.get('content', '')[:50] + ('...' if len(first_msg.get('content', '')) > 50 else '')
        
        print(f"\n  [{i}/{len(conversations)}] 导入对话: {user_content[:40]}...")
        
        try:
            # 使用 add_memory_from_messages 导入
            # 注意：使用 base_tags 而不是 tags 参数
            node = ms.add_memory_from_messages(
                messages=session,
                source="cli-import",
                base_tags=["cli-import", start_date[:7]]  # 标签如 "2026-04"
            )
            if node:
                imported_count += 1
                print(f"       ✅ 成功导入: {node.url}")
            else:
                print(f"       ⚠️ 内容为空，跳过")
        except Exception as e:
            print(f"       ❌ 导入失败: {e}")
    
    print(f"\n5. 保存索引...")
    try:
        # 保存索引
        if hasattr(ms, '_save_index'):
            ms._save_index()
        ms.save()
        print(f"   索引已保存")
    except Exception as e:
        print(f"   警告: 索引保存时出错: {e}")
    
    return imported_count


def main():
    input_file, start_date, end_date = parse_args()
    
    # 检查文件是否存在
    if not Path(input_file).exists():
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    # 加载消息
    messages = load_messages_by_date(input_file, start_date, end_date)
    
    if not messages:
        print(f"未找到 {start_date} 到 {end_date} 之间的消息")
        sys.exit(0)
    
    # 分组为对话
    conversations = group_into_conversations(messages)
    
    if not conversations:
        print("没有有效的对话可以导入")
        sys.exit(0)
    
    # 确认导入
    print(f"\n准备导入 {len(conversations)} 组对话到 Memora")
    response = input("确认导入? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        sys.exit(0)
    
    # 导入到 Memora
    imported = import_to_memora(conversations, start_date, end_date)
    
    print(f"\n✅ 导入完成!")
    print(f"   成功导入: {imported}/{len(conversations)} 组对话")
    print(f"   日期范围: {start_date} 到 {end_date}")
    print(f"   数据来源: {input_file}")


if __name__ == "__main__":
    main()
