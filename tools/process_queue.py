# -*- coding: utf-8 -*-
"""
Memory Queue Processor - 处理待导入的记忆队列
由 Heartbeat 每30分钟调用一次
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime

MEMORY_SYSTEM_PATH = Path("/Users/rama/.nanobot/workspace/Memora")
QUEUE_FILE = MEMORY_SYSTEM_PATH / "_pending_queue.jsonl"


def process_queue():
    """处理待导入的记忆队列"""
    if not QUEUE_FILE.exists():
        print("[Queue] 队列为空，无需处理")
        return 0
    
    # 读取队列
    with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if not lines:
        print("[Queue] 队列为空，无需处理")
        return 0
    
    print(f"[Queue] 发现 {len(lines)} 条待导入记忆")
    
    imported = 0
    failed = 0
    
    for line in lines:
        try:
            data = json.loads(line)
            
            # 直接创建 markdown 文件（绕过复杂的 import 系统）
            node_id = f"{datetime.now().strftime('%Y%m%d%H%M')}-{os.urandom(4).hex()[:8]}"
            date_path = datetime.now().strftime('%Y/%m/%d')
            memory_dir = MEMORY_SYSTEM_PATH / "data" / date_path
            memory_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = memory_dir / f"{node_id}.md"
            
            # 写入 markdown
            title = data.get('title', 'Untitled')
            tags = data.get('tags', [])
            content = data.get('content', '')
            
            md_content = f"""---
id: {node_id}
timestamp: {datetime.now().isoformat()}
title: {title}
tags: {tags}
source: {data.get('source', 'unknown')}
pagerank: 1.0
---

{content}
"""
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            print(f"  ✅ 导入: {title[:40]}... (id={node_id})")
            imported += 1
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)[:50]}")
            failed += 1
    
    # 清空队列
    QUEUE_FILE.unlink()
    print(f"[Queue] 清理完成")
    
    print(f"\n[Queue] 导入完成: {imported} 成功, {failed} 失败")
    
    # ===== 重建记忆图谱和PageRank =====
    if imported > 0:
        print("\n[Graph] 重建记忆图谱...")
        try:
            # 使用 subprocess 调用 build_graph.py 脚本
            import subprocess
            result = subprocess.run(
                ["python3", "build_graph.py"],
                cwd=MEMORY_SYSTEM_PATH,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"  ⚠️ stderr: {result.stderr[:200]}")
            else:
                print(f"  ✅ 图谱重建完成")
                    
        except Exception as e:
            print(f"  ❌ 图谱重建失败: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
    
    return imported


if __name__ == "__main__":
    process_queue()
