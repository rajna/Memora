#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import April 5, 2026 conversations to Memora (batch mode with progress)"""
import json
import sys
import time
import os
from datetime import datetime

sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem

input_file = "/Users/rama/.nanobot/workspace/sessions/cli_direct.jsonl"
progress_file = "/tmp/import_april5_progress.txt"

def is_april_5(ts):
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.year == 2026 and dt.month == 4 and dt.day == 5
    except:
        return False

# Load progress
start_idx = 0
if os.path.exists(progress_file):
    try:
        with open(progress_file, 'r') as f:
            start_idx = int(f.read().strip())
        print(f"Resuming from session {start_idx}")
    except:
        pass

print("1. Parsing April 5 sessions...")
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
            
            is_d = is_april_5(ts)
            
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

print(f"   Found {len(conversations)} sessions")

if not conversations:
    print("   No data found")
    sys.exit(0)

if start_idx >= len(conversations):
    print("All sessions already imported!")
    sys.exit(0)

print(f"\n2. Importing sessions {start_idx+1}-{len(conversations)}...")
print(f"   (Processing in batches of 20)")
ms = MemorySystem()
imported = 0
batch_size = 20
start = time.time()

for i in range(start_idx, len(conversations)):
    session = conversations[i]
    try:
        node = ms.add_memory_from_messages(
            messages=session,
            source="cli-april-5",
            base_tags=["cli-import", "2026-04-05"]
        )
        if node:
            imported += 1
    except Exception as e:
        print(f"   FAIL {i+1}: {str(e)[:50]}")
    
    # Save progress every session
    with open(progress_file, 'w') as f:
        f.write(str(i + 1))
    
    # Print progress every batch
    if (i + 1) % batch_size == 0 or i == len(conversations) - 1:
        elapsed = time.time() - start
        print(f"   Progress: {i+1}/{len(conversations)} sessions ({elapsed:.1f}s)")
        sys.stdout.flush()

elapsed = time.time() - start
print(f"\n   Imported {imported}/{len(conversations) - start_idx} nodes in {elapsed:.1f}s")
print("\n✅ Done!")

# Clean up progress file
if os.path.exists(progress_file):
    os.remove(progress_file)
