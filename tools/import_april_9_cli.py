#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import April 9, 2026 conversations from cli_direct.jsonl to Memora
"""
import json
import sys
from datetime import datetime

sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem

input_file = "/Users/rama/.nanobot/workspace/sessions/cli_direct.jsonl"

def is_april_9(timestamp_str):
    """Check if timestamp is 2026-04-09"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.year == 2026 and dt.month == 4 and dt.day == 9
    except:
        return False

# Load all April 9 sessions
conversations = []
current_session = []
in_april_9 = False

print("1. Loading April 9 session data...")
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
                
            msg_is_april_9 = is_april_9(timestamp)
            
            if role == 'user':
                if current_session and in_april_9:
                    conversations.append(current_session)
                
                current_session = [msg]
                in_april_9 = msg_is_april_9
                
            elif role in ('assistant', 'tool') and current_session:
                current_session.append(msg)
                if msg_is_april_9:
                    in_april_9 = True
                    
        except Exception as e:
            pass

if current_session and in_april_9:
    conversations.append(current_session)

print("   Found {} April 9 conversation groups".format(len(conversations)))

if not conversations:
    print("   No April 9 conversations found, exiting")
    sys.exit(0)

# Initialize MemorySystem
print("\n2. Initializing MemorySystem...")
ms = MemorySystem()

# Import sessions
print("\n3. Importing April 9 sessions...")
imported = 0
failed = []
for i, session in enumerate(conversations, 1):
    try:
        node = ms.add_memory_from_messages(
            messages=session,
            source="cli-april-9",
            base_tags=["cli-import", "2026-04-09"]
        )
        if node:
            imported += 1
            if i % 20 == 0 or i == len(conversations):
                print("   Progress: {}/{} - {}".format(i, len(conversations), node.id))
        else:
            failed.append(i)
            print("   FAIL {}/{} (node creation failed)".format(i, len(conversations)))
    except Exception as e:
        failed.append(i)
        print("   FAIL {}/{}: {}".format(i, len(conversations), str(e)[:80]))

print("\n   Import complete: {} success, {} failed".format(imported, len(failed)))

# Save
print("\n4. Saving index...")
ms._save_index()

print("\n✅ Done! Imported {} April 9 nodes".format(imported))
