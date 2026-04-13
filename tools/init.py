#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory System Initialization Script
网页记忆系统初始化脚本
"""
import os
import sys

def init_directories():
    """初始化目录结构"""
    dirs = [
        "src",
        "web",
        "tests",
        "data",
        "embeddings",
        "index",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ {d}/")
    
    # 在workspace根目录创建Memora链接
    workspace = "/Users/rama/.nanobot/workspace"
    target = os.path.join(workspace, "Memora")
    
    print(f"\nMemory System initialized at: {target}")
    print(f"  Data: {target}/data")
    print(f"  Embeddings: {target}/embeddings")

def check_dependencies():
    """检查依赖"""
    print("\nChecking dependencies...")
    
    deps = [
        ("python-frontmatter", "python-frontmatter"),
        ("networkx", "networkx"),
        ("numpy", "numpy"),
        ("sentence-transformers", "sentence-transformers (optional)"),
        ("flask", "flask (for web viewer)"),
    ]
    
    all_ok = True
    for module, name in deps:
        try:
            __import__(module.replace("-", "_"))
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - not installed")
            all_ok = False
    
    if not all_ok:
        print("\nInstall missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return all_ok

if __name__ == "__main__":
    print("=" * 50)
    print("Memory System Initialization")
    print("=" * 50)
    
    init_directories()
    check_dependencies()
    
    print("\n" + "=" * 50)
    print("Quick Start:")
    print("  python tests/test_memory_system.py")
    print("  python web/viewer.py")
    print("=" * 50)
