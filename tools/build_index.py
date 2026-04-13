#!/usr/bin/env python3
"""构建索引文件供 benchmark 使用"""

import pickle
import numpy as np
from pathlib import Path


def build_index():
    """从 embeddings 目录构建索引"""
    embedding_dir = Path("/Users/rama/.nanobot/workspace/Memora/embeddings")
    index_dir = Path("/Users/rama/.nanobot/workspace/Memora/index")
    index_dir.mkdir(exist_ok=True)
    
    # 加载所有 embeddings
    embeddings = []
    node_ids = []
    
    for npy_file in sorted(embedding_dir.glob("*.npy")):
        node_id = npy_file.stem
        vec = np.load(npy_file)
        embeddings.append(vec)
        node_ids.append(node_id)
    
    if not embeddings:
        print("⚠️  没有找到 embedding 文件")
        return
    
    # 转换为 numpy 数组
    embeddings = np.array(embeddings)
    
    # 保存索引
    index_file = index_dir / "embeddings.pkl"
    with open(index_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'node_ids': node_ids,
        }, f)
    
    print(f"✅ 构建索引: {len(node_ids)} 个节点")
    print(f"   形状: {embeddings.shape}")
    print(f"   保存到: {index_file}")


if __name__ == "__main__":
    build_index()
