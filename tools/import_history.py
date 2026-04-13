#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import existing HISTORY.md into Memory System
将现有HISTORY.md导入记忆系统
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import re
from datetime import datetime
from pathlib import Path

# 修复相对导入
import src.models as models
import src.config as config
sys.modules['models'] = models
sys.modules['config'] = config

from src.memory_system import MemorySystem


def parse_history_file(history_path: str):
    """
    解析HISTORY.md文件
    
    格式:
    [2026-04-08 17:24] User proposed a new skill idea...
    [2026-04-08 17:24-17:30] Multi-line entry...
    """
    with open(history_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配时间戳模式
    # [YYYY-MM-DD HH:MM] 或 [YYYY-MM-DD HH:MM-HH:MM]
    pattern = r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?:-\d{2}:\d{2})?)\]\s*(.*?)(?=\[\d{4}-\d{2}-\d{2}|\Z)'
    
    entries = []
    for match in re.finditer(pattern, content, re.DOTALL):
        timestamp_str = match.group(1)
        entry_text = match.group(2).strip()
        
        # 解析时间戳
        try:
            # 处理跨时段 [2026-04-08 17:24-17:30]
            if '-' in timestamp_str and timestamp_str.count(':') == 2:
                # 只取开始时间
                timestamp_str = timestamp_str.split('-')[0]
            
            timestamp = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M")
        except ValueError:
            print(f"Warning: Could not parse timestamp: {timestamp_str}")
            continue
        
        if entry_text:
            entries.append({
                'timestamp': timestamp,
                'content': entry_text,
                'title': entry_text[:50] + '...' if len(entry_text) > 50 else entry_text
            })
    
    return entries


def import_to_memory_system(entries: list, ms: MemorySystem, batch_size: int = 10):
    """
    导入条目到记忆系统
    """
    print(f"Importing {len(entries)} entries...")
    
    nodes = []
    for i, entry in enumerate(entries):
        # 提取标签（从内容中识别关键词）
        tags = extract_tags(entry['content'])
        
        node = ms.add_memory(
            content=entry['content'],
            title=entry['title'],
            tags=tags,
            source="cli",
        )
        nodes.append(node)
        
        if (i + 1) % batch_size == 0:
            print(f"  Progress: {i + 1}/{len(entries)}")
    
    print(f"\n✓ Imported {len(nodes)} entries")
    return nodes


def extract_tags(content: str) -> list:
    """
    从内容中提取标签
    """
    tags = []
    content_lower = content.lower()
    
    # 项目关键词
    if any(k in content_lower for k in ['vrm', 'mmd', 'motion capture', '动捕']):
        tags.append('vrm')
    if any(k in content_lower for k in ['memory', 'pagerank', 'memory system']):
        tags.append('Memora')
    if any(k in content_lower for k in ['skill', '技能']):
        tags.append('skill')
    if any(k in content_lower for k in ['novel', '小说', '灵痕', '修仙']):
        tags.append('novel')
    if any(k in content_lower for k in ['electron', 'nw.js', 'browser']):
        tags.append('electron')
    if any(k in content_lower for k in ['claude', 'minimax', 'api']):
        tags.append('ai-api')
    
    # 技术关键词
    if any(k in content_lower for k in ['mediapipe', 'holistic', 'pose']):
        tags.append('mediapipe')
    if any(k in content_lower for k in ['three.js', '3d', 'webgl']):
        tags.append('threejs')
    
    return tags


def build_links(nodes: list, ms: MemorySystem, time_window_hours: int = 2):
    """
    基于时间窗口自动建立链接
    """
    print("\nBuilding temporal links...")
    
    link_count = 0
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            time_diff = abs((node1.created - node2.created).total_seconds())
            if time_diff <= time_window_hours * 3600:
                # 时间相近，建立双向链接
                if node2.url not in node1.links:
                    node1.links.append(node2.url)
                if node1.url not in node2.links:
                    node2.links.append(node1.url)
                link_count += 2
    
    # 保存更新
    for node in nodes:
        ms.storage.save(node)
    
    print(f"✓ Created {link_count} temporal links")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Import HISTORY.md to Memory System')
    parser.add_argument('--history', default='/Users/rama/.nanobot/workspace/memory/HISTORY.md',
                        help='Path to HISTORY.md')
    parser.add_argument('--skip-import', action='store_true',
                        help='Skip import, only build graph')
    
    args = parser.parse_args()
    
    ms = MemorySystem()
    
    if not args.skip_import:
        if not os.path.exists(args.history):
            print(f"Error: {args.history} not found")
            sys.exit(1)
        
        # 解析HISTORY.md
        entries = parse_history_file(args.history)
        print(f"Found {len(entries)} entries in {args.history}")
        
        # 导入到记忆系统
        nodes = import_to_memory_system(entries, ms)
        
        # 建立时间链接
        build_links(nodes, ms)
    
    # 构建图谱并计算PageRank
    print("\nBuilding graph and calculating PageRank...")
    scores = ms.build_graph(auto_link=False)
    
    # 显示统计
    stats = ms.stats()
    print(f"\n{'='*50}")
    print(f"Memory System Statistics:")
    print(f"  Total memories: {stats['total_nodes']}")
    print(f"  Total tags: {stats['total_tags']}")
    print(f"  Avg PageRank: {stats['avg_pagerank']:.4f}")
    print(f"{'='*50}")
