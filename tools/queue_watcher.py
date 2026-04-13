# -*- coding: utf-8 -*-
"""
Queue Watcher - 30秒循环监控队列
测试用，Ctrl+C 停止
"""
import time
import subprocess
import sys
from pathlib import Path

print("🧠 Queue Watcher 启动 (30秒间隔)")
print("按 Ctrl+C 停止\n")

count = 0
while True:
    try:
        # 执行导入
        result = subprocess.run(
            [sys.executable, "process_queue.py"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            print(result.stdout)
            count += 1
            print(f"[第 {count} 次执行] ---")
        
        # 30秒间隔
        time.sleep(30)
        
    except KeyboardInterrupt:
        print(f"\n\n✅ 共执行 {count} 次，退出")
        break
    except Exception as e:
        print(f"❌ 错误: {e}")
        time.sleep(30)
