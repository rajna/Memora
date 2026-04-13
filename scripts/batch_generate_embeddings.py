#!/usr/bin/env python3
"""
批量为 Memora 的记忆节点生成向量(embedding)
"""
import os
import re
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# 配置
NODES_DIR = 'data/2026'
EMBEDDINGS_DIR = 'data/2026'  # embedding 保存在同级目录
MODEL_NAME = 'all-MiniLM-L6-v2'

def parse_frontmatter(content):
    """解析 markdown 的 frontmatter"""
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return None, content
    try:
        fm = yaml.safe_load(match.group(1))
        body = content[match.end():]
        return fm, body
    except:
        return None, content

def extract_text(fm, body):
    """从节点提取可向量化文本"""
    parts = []
    if fm:
        if fm.get('title'):
            parts.append(fm['title'])
        if fm.get('tags'):
            if isinstance(fm['tags'], list):
                parts.append(' '.join(fm['tags']))
            else:
                parts.append(str(fm['tags']))
    
    # 从 body 提取对话
    lines = body.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('[用户]') or line.startswith('[AI]'):
            parts.append(line)
    
    return ' '.join(parts) if parts else body[:500]

def main():
    start_time = time.time()
    
    print(f"🚀 加载模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   Embedding 维度: {embedding_dim}")
    
    # 收集所有节点
    nodes = []
    for root, dirs, files in os.walk(NODES_DIR):
        for f in sorted(files):
            if f.endswith('.md'):
                path = os.path.join(root, f)
                nodes.append(path)
    
    total = len(nodes)
    print(f"\n📊 找到 {total} 个节点\n")
    
    success = 0
    failed = 0
    failed_list = []
    
    # 批量处理
    batch_size = 32
    for i in tqdm(range(0, total, batch_size), desc="生成embedding"):
        batch_paths = nodes[i:i+batch_size]
        batch_texts = []
        batch_ids = []
        
        for path in batch_paths:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            fm, body = parse_frontmatter(content)
            text = extract_text(fm, body)
            batch_texts.append(text)
            
            # 从文件名提取 id
            node_id = os.path.splitext(os.path.basename(path))[0]
            batch_ids.append(node_id)
        
        # 批量编码
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        
        # 保存
        for j, node_id in enumerate(batch_ids):
            emb_path = os.path.join(EMBEDDINGS_DIR, f"{node_id}.npy")
            try:
                np.save(emb_path, embeddings[j])
                success += 1
            except Exception as e:
                failed += 1
                failed_list.append((node_id, str(e)))
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ 完成!")
    print(f"   总节点数: {total}")
    print(f"   成功: {success}")
    print(f"   失败: {failed}")
    print(f"   总耗时: {elapsed:.2f}秒")
    print(f"   速度: {total/elapsed:.1f} 节点/秒")
    
    if failed_list:
        print(f"\n⚠️ 失败列表:")
        for node_id, err in failed_list[:10]:
            print(f"   {node_id}: {err}")

if __name__ == '__main__':
    main()
