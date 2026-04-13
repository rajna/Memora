#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch import April 8 conversations (optimized)"""
import json
import sys
from datetime import datetime

sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem

input_file = "/Users/rama/.nanobot/workspace/sessions/cli_direct.jsonl"

def is_april_8(ts):
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.year == 2026 and dt.month == 4 and dt.day == 8
    except:
        return False

# Parse sessions first (fast)
print("Parsing April 8 sessions...")
conversations = []
current = []
in_date = False

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
            
            is_date = is_april_8(ts)
            
            if role == 'user':
                if current and in_date:
                    conversations.append(current)
                current = [msg]
                in_date = is_date
            elif role in ('assistant', 'tool') and current:
                current.append(msg)
                if is_date:
                    in_date = True
        except:
            pass

if current and in_date:
    conversations.append(current)

print("Found {} sessions".format(len(conversations)))

# Import in batches
print("\nImporting...")
ms = MemorySystem()
imported = 0

for i, session in enumerate(conversations, 1):
    try:
        node = ms.add_memory_from_messages(
            messages=session,
            source="cli-april-8",
            base_tags=["cli-import", "2026-04-08"]
        )
        if node:
            imported += 1
    except Exception as e:
        print("Error at {}: {}".format(i, str(e)[:50]))

ms._save_index()
print("\n✅ Done! Imported {}/{}".format(imported, len(conversations)))
