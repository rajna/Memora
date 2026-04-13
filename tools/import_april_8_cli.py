#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import April 8, 2026 conversations from cli_direct.jsonl to Memora"""
import json
import sys
from datetime import datetime

sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem

input_file = "/Users/rama/.nanobot/workspace/sessions/cli_direct.jsonl"

def is_april_8(timestamp_str):
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.year == 2026 and dt.month == 4 and dt.day == 8
    except:
        return False

conversations = []
current_session = []
in_april_8 = False

print("1. Loading April 8 session data...")
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            role = msg.get('role', '')
            timestamp = msg.get('timestamp', '')
            
            if role == '_type':
                continue
                
            msg_is_april_8 = is_april_8(timestamp)
            
            if role == 'user':
                if current_session and in_april_8:
                    conversations.append(current_session)
                current_session = [msg]
                in_april_8 = msg_is_april_8
            elif role in ('assistant', 'tool') and current_session:
                current_session.append(msg)
                if msg_is_april_8:
                    in_april_8 = True
        except:
            pass

if current_session and in_april_8:
    conversations.append(current_session)

print("   Found {} April 8 conversation groups".format(len(conversations)))

if not conversations:
    print("   No April 8 conversations found")
    sys.exit(0)

print("\n2. Initializing MemorySystem...")
ms = MemorySystem()

print("\n3. Importing April 8 sessions...")
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
            if i % 25 == 0 or i == len(conversations):
                print("   Progress: {}/{} - {}".format(i, len(conversations), node.id))
    except Exception as e:
        print("   FAIL {}/{}: {}".format(i, len(conversations), str(e)[:60]))

print("\n   Import complete: {} success".format(imported))
print("\n4. Saving index...")
ms._save_index()
print("\n✅ Done! Imported {} April 8 nodes".format(imported))
