#!/usr/bin/env python3
"""
为缺少嵌入的节点重新生成向量
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.memory_system import MemorySystem
from src.embeddings import get_embedding_manager
from src.storage import MemoryStorage
from src.config import MEMORY_DIR

def main():
    print("🔧 重新生成缺失的嵌入...")
    
    ms = MemorySystem()
    storage = MemoryStorage(MEMORY_DIR)
    emb_mgr = get_embedding_manager()
    
    # 找出没有嵌入的节点
    nodes = storage.get_all()
    missing = []
    for node in nodes:
        emb = emb_mgr.load_embedding(node.id)
        if emb is None:
            missing.append(node)
    
    print(f"发现 {len(missing)} 个节点缺少嵌入")
    
    # 生成嵌入
    for i, node in enumerate(missing):
        text = f"{node.title}\n{node.content}" if node.title else node.content
        if text.strip():
            emb = emb_mgr.encode_single(text)
            emb_mgr.save_embedding(node.id, emb)
            node.embedding_file = f"{node.id}.npy"
            storage.save(node)
            print(f"  [{i+1}/{len(missing)}] {node.id}: {node.title[:40] if node.title else '(no title)'}...")
    
    print(f"\n✅ 完成！已生成 {len(missing)} 个嵌入")

if __name__ == "__main__":
    main()
