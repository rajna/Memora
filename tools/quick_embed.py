#!/usr/bin/env python3
"""快速生成少量 embedding 用于测试"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora/src')

from memory_system import MemorySystem
from storage import MemoryStorage
from embeddings import get_embedding_manager
from config import MEMORY_DIR

def main():
    print("🔧 生成前20个节点的嵌入...")
    
    storage = MemoryStorage(MEMORY_DIR)
    emb_mgr = get_embedding_manager()
    
    nodes = storage.get_all()[:20]  # 只取前20个
    
    for i, node in enumerate(nodes):
        text = f"{node.title}\n{node.content}" if node.title else node.content
        if text.strip():
            print(f"  [{i+1}/20] {node.id[:20]}...")
            emb = emb_mgr.encode_single(text)
            emb_mgr.save_embedding(node.id, emb)
            node.embedding_file = f"{node.id}.npy"
            storage.save(node)
    
    print(f"\n✅ 完成！生成了 20 个嵌入")

if __name__ == "__main__":
    main()
