#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import April 4, 2026 conversations to Memora"""
import json
import sys
import time
from datetime import datetime

sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem

input_file = "/Users/rama/.nanobot/workspace/sessions/cli_direct.jsonl"

def is_april_4(ts):
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.year == 2026 and dt.month == 4 and dt.day == 4
    except:
        return False

conversations = []
current = []
in_date = False

print("1. Parsing April 4 sessions...")
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            role = msg.get('role', '')
            ts = msg.get('timestamp', '')
            
            if role == '_type':
                continue
            
            is_d = is_april_4(ts)
            
            if role == 'user':
                if current and in_date:
                    conversations.append(current)
                current = [msg]
                in_date = is_d
            elif role in ('assistant', 'tool') and current:
                current.append(msg)
                if is_d:
                    in_date = True
        except:
            pass

if current and in_date:
    conversations.append(current)

print("   Found {} sessions".format(len(conversations)))

if not conversations:
    print("   No data found")
    sys.exit(0)

# 按消息数量排序，优先处理小的session
conversations.sort(key=lambda s: len(s))

print("\n2. Importing (processing {} sessions)...".format(len(conversations)))
print("   Sessions sorted by size (smallest first)")
print("   (This may take several minutes for large datasets)")
print()

ms = MemorySystem()
imported = 0
failed = 0
start = time.time()
last_progress = 0
batch_size = 10

for i, session in enumerate(conversations, 1):
    try:
        node = ms.add_memory_from_messages(
            messages=session,
            source="cli-april-4",
            base_tags=["cli-import", "2026-04-04"]
        )
        if node:
            imported += 1
    except Exception as e:
        failed += 1
        if failed <= 3:  # 只显示前3个错误
            print("   FAIL {}: {}".format(i, str(e)[:50]))
    
    # 每 batch_size 个输出一次进度
    if i % batch_size == 0 or i == len(conversations):
        elapsed = time.time() - start
        progress = i / len(conversations) * 100
        eta = (elapsed / i) * (len(conversations) - i) if i > 0 else 0
        print("   Progress: {}/{} ({:.1f}%) - Imported: {} - ETA: {:.0f}s".format(
            i, len(conversations), progress, imported, eta))
        sys.stdout.flush()

elapsed = time.time() - start
print("\n   Imported {}/{} nodes in {:.1f}s".format(imported, len(conversations), elapsed))
if failed > 0:
    print("   Failed: {}".format(failed))
print("\n✅ Done!")
