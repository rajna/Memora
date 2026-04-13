#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import April 3, 2026 conversations to Memora"""
import json
import sys
import time
from datetime import datetime

sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem

input_file = "/Users/rama/.nanobot/workspace/sessions/cli_direct.jsonl"

def is_april_3(ts):
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.year == 2026 and dt.month == 4 and dt.day == 3
    except:
        return False

conversations = []
current = []
in_date = False

print("1. Parsing April 3 sessions...")
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
            
            is_d = is_april_3(ts)
            
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

print("\n2. Importing (large dataset, please wait)...")
ms = MemorySystem()
imported = 0
start = time.time()

for i, session in enumerate(conversations, 1):
    try:
        node = ms.add_memory_from_messages(
            messages=session,
            source="cli-april-3",
            base_tags=["cli-import", "2026-04-03"]
        )
        if node:
            imported += 1
    except Exception as e:
        print("   FAIL {}: {}".format(i, str(e)[:50]))

elapsed = time.time() - start
print("\n   Imported {}/{} nodes in {:.1f}s".format(imported, len(conversations), elapsed))
print("\n✅ Done!")
