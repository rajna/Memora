#!/usr/bin/env python3
"""
批量重建所有节点的 embedding (切换模型后使用)
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

from pathlib import Path
from src.storage import MemoryStorage
from src.embeddings import EmbeddingManager

def main():
    print("=" * 60)
    print("Rebuilding All Embeddings")
    print("=" * 60)
    
    # 初始化
    from src.config import MEMORY_DIR
    storage = MemoryStorage(base_dir=MEMORY_DIR)
    emb_mgr = EmbeddingManager()  # 使用 config.py 中的新模型
    
    # 获取所有节点
    nodes = storage.get_all()
    print(f"\n📊 Total nodes: {len(nodes)}")
    print(f"🤖 Model: {emb_mgr.model_name}")
    print(f"📐 Dimension: {emb_mgr._dim}")
    
    # 备份旧 embeddings
    import shutil
    from datetime import datetime
    backup_dir = Path("embeddings-backup") / f"backup_{datetime.now():%Y%m%d_%H%M}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    old_emb_dir = Path("embeddings")
    if old_emb_dir.exists():
        for f in old_emb_dir.glob("*.npy"):
            shutil.copy(f, backup_dir / f.name)
        print(f"\n💾 Backup created: {backup_dir}")
    
    # 批量生成新 embedding
    print(f"\n🔨 Generating new embeddings...")
    
    success = 0
    failed = 0
    
    for i, node in enumerate(nodes, 1):
        try:
            # 组合文本：标题 + 内容 + 标签
            text_parts = []
            if node.title:
                text_parts.append(node.title)
            if node.content:
                text_parts.append(node.content[:1000])  # 限制长度
            if node.tags:
                text_parts.append(" ".join(node.tags))
            
            text = "\n".join(text_parts)
            
            # 生成 embedding
            embedding = emb_mgr.encode_single(text)
            
            # 保存
            emb_mgr.save_embedding(node.id, embedding)
            
            # 更新节点元数据
            node.embedding_file = f"embeddings/{node.id}.npy"
            storage.save(node)  # 保存回文件
            
            success += 1
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(nodes)} ({success} success, {failed} failed)")
                
        except Exception as e:
            failed += 1
            print(f"  ⚠️ Failed: {node.id} - {e}")
    
    print(f"\n✅ Done!")
    print(f"   Success: {success}")
    print(f"   Failed: {failed}")
    print(f"\n💾 New embeddings saved to: embeddings/")
    
    # 显示文件大小变化
    import subprocess
    result = subprocess.run(["du", "-sh", "embeddings/"], capture_output=True, text=True)
    print(f"   Size: {result.stdout.strip()}")

if __name__ == "__main__":
    main()
