# -*- coding: utf-8 -*-
"""
Build Memory Graph and Calculate PageRank
重建记忆图谱并计算 PageRank
"""
import sys
import os

# 设置环境变量
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 使用 -m 方式运行
import subprocess
result = subprocess.run(
    [sys.executable, "-m", "src.build_graph_cli"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    capture_output=False
)
sys.exit(result.returncode)
