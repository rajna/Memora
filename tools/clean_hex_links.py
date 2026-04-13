#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 Memora 中被误识别为链接的8位十六进制节点ID
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.memory_system import MemorySystem


def is_hex_id(url: str) -> bool:
    """检查URL最后部分是否是8位十六进制"""
    if not url:
        return False
    parts = url.strip('/').split('/')
    if not parts:
        return False
    last = parts[-1]
    return len(last) == 8 and all(c in '0123456789abcdef' for c in last.lower())


def clean_node_links(node, dry_run: bool = True) -> int:
    """
    清理节点的链接
    返回清理的链接数量
    """
    original_links = node.links.copy()
    cleaned_links = [url for url in node.links if not is_hex_id(url)]
    
    removed_count = len(original_links) - len(cleaned_links)
    
    if removed_count > 0 and not dry_run:
        node.links = cleaned_links
        print(f"  清理 {node.id}: 移除 {removed_count} 个十六进制链接")
        for url in original_links:
            if is_hex_id(url):
                print(f"    - {url}")
    
    return removed_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='清理十六进制链接')
    parser.add_argument('--dry-run', action='store_true', 
                        help='仅预览，不实际修改')
    parser.add_argument('--apply', action='store_true',
                        help='实际执行清理')
    args = parser.parse_args()
    
    dry_run = not args.apply
    
    print("🧹 Memory System 链接清理工具")
    print("=" * 50)
    print(f"模式: {'预览' if dry_run else '执行'}")
    print()
    
    ms = MemorySystem()
    nodes = ms.storage.get_all()
    
    print(f"共 {len(nodes)} 个节点")
    print()
    
    total_cleaned = 0
    affected_nodes = 0
    
    for node in nodes:
        try:
            cleaned = clean_node_links(node, dry_run)
            if cleaned > 0:
                affected_nodes += 1
                total_cleaned += cleaned
                
                if not dry_run:
                    # 保存修改后的节点
                    ms.storage.save(node)
        except Exception as e:
            print(f"  ⚠️ 跳过节点 {node.id}: {str(e)[:50]}")
    
    print()
    print("=" * 50)
    print(f"统计:")
    print(f"  受影响节点: {affected_nodes}")
    print(f"  清理链接数: {total_cleaned}")
    
    if dry_run and total_cleaned > 0:
        print()
        print(f"💡 使用 --apply 参数执行实际清理")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
