#!/usr/bin/env python3
"""
增量重建 Embedding 脚本 V2
修复：使用 id 字段作为标识符（兼容 LME 节点）
"""
import os
import sys
import time
import json
import gc
import re
import frontmatter
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 未安装")
    sys.exit(1)

# 配置
NODES_DIR = Path("data/memory-nodes")
EMBEDDINGS_DIR = Path("embeddings")
PROGRESS_FILE = EMBEDDINGS_DIR / ".rebuild_progress_v2.json"
LOG_FILE = Path("logs") / f"incremental_rebuild_v2_{datetime.now():%m%d_%H%M}.log"
BATCH_SIZE = 100
PAUSE_SECONDS = 2

def log(msg):
    """记录日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')

def get_node_id(post):
    """获取节点的唯一标识符"""
    # 优先使用 id，其次是 url
    node_id = post.get('id', '')
    if not node_id:
        node_id = post.get('url', '')
    return node_id

def find_nodes_without_embedding():
    """找出所有缺少 embedding 的节点"""
    existing_npy = set(f.name for f in EMBEDDINGS_DIR.glob("*.npy"))
    
    nodes_to_process = []
    total = 0
    
    for node_file in NODES_DIR.rglob("*.md"):
        total += 1
        try:
            post = frontmatter.load(node_file)
            node_id = get_node_id(post)
            
            if not node_id:
                log(f"警告: 节点 {node_file} 没有 id 或 url")
                continue
            
            embedding_file = post.get("embedding_file")
            
            if not embedding_file:
                # 没有 embedding_file 字段
                nodes_to_process.append({
                    'id': node_id,
                    'file': str(node_file)
                })
            else:
                # 检查文件是否存在
                npy_name = Path(embedding_file).name
                if npy_name not in existing_npy:
                    nodes_to_process.append({
                        'id': node_id,
                        'file': str(node_file)
                    })
        except Exception as e:
            log(f"加载节点失败 {node_file}: {e}")
    
    return nodes_to_process, total

def generate_embedding(text: str, model) -> np.ndarray:
    """生成 embedding 向量"""
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding.astype(np.float32)

def process_node(node_info, model):
    """处理单个节点"""
    try:
        post = frontmatter.load(node_info['file'])
        
        # 提取文本内容
        content = post.content
        title = post.get('title', '')
        tags = post.get('tags', [])
        
        # 组合文本
        text_parts = []
        if title:
            text_parts.append(title)
        if tags:
            text_parts.append(' '.join(str(t) for t in tags))
        text_parts.append(content)
        
        full_text = '\n'.join(text_parts)
        
        # 截断文本（如果需要）
        max_length = 8000
        if len(full_text) > max_length:
            full_text = full_text[:max_length]
        
        # 生成 embedding
        embedding = generate_embedding(full_text, model)
        
        # 保存 embedding 文件
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        random_suffix = os.urandom(4).hex()
        npy_filename = f"{timestamp}-{random_suffix}.npy"
        npy_path = EMBEDDINGS_DIR / npy_filename
        
        np.save(npy_path, embedding)
        
        # 更新节点文件
        post.metadata['embedding_file'] = f"embeddings/{npy_filename}"
        with open(node_info['file'], 'w') as f:
            f.write(frontmatter.dumps(post))
        
        return True, npy_filename
    except Exception as e:
        return False, str(e)

def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    log("=" * 60)
    log("开始增量重建 Embedding V2")
    log("=" * 60)
    
    # 加载进度
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        processed = set(progress.get('processed', []))
        failed = progress.get('failed', [])
    else:
        processed = set()
        failed = []
        progress = {'processed': [], 'failed': [], 'start_time': datetime.now().isoformat()}
    
    log(f"已处理过的节点: {len(processed)} 个")
    log(f"之前失败的节点: {len(failed)} 个")
    
    # 找出需要处理的节点
    all_missing_nodes, total_nodes = find_nodes_without_embedding()
    
    # 过滤掉已处理的
    nodes_to_process = [n for n in all_missing_nodes if n['id'] not in processed]
    
    log(f"总节点数: {total_nodes}")
    log(f"当前缺少 embedding: {len(all_missing_nodes)} 个")
    log(f"实际需要处理: {len(nodes_to_process)} 个")
    
    if not nodes_to_process:
        log("✓ 所有节点都已有 embedding，无需处理！")
        return
    
    # 初始化模型
    log("加载 embedding 模型...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    log("模型加载完成")
    
    # 分批处理
    total_to_process = len(nodes_to_process)
    processed_count = 0
    failed_count = 0
    
    for i in range(0, total_to_process, BATCH_SIZE):
        batch = nodes_to_process[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total_to_process + BATCH_SIZE - 1) // BATCH_SIZE
        
        log(f"\n处理批次 {batch_num}/{total_batches} ({len(batch)} 个节点)")
        
        for node_info in batch:
            success, result = process_node(node_info, model)
            
            if success:
                processed.add(node_info['id'])
                processed_count += 1
            else:
                failed.append({'id': node_info['id'], 'error': result})
                failed_count += 1
                log(f"  失败: {node_info['id']} - {result}")
        
        # 保存进度
        progress['processed'] = list(processed)
        progress['failed'] = failed
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
        
        log(f"  批次完成: 成功 {len(batch) - (1 if failed_count > 0 else 0)}/{len(batch)}")
        log(f"  总体进度: {processed_count}/{total_to_process} ({processed_count/total_to_process*100:.1f}%)")
        
        # 暂停避免系统终止
        if i + BATCH_SIZE < total_to_process:
            log(f"  暂停 {PAUSE_SECONDS} 秒...")
            time.sleep(PAUSE_SECONDS)
            gc.collect()
    
    log("\n" + "=" * 60)
    log("增量重建完成！")
    log(f"成功: {processed_count} 个")
    log(f"失败: {failed_count} 个")
    log(f"总 .npy 文件: {len(list(EMBEDDINGS_DIR.glob('*.npy')))} 个")
    log("=" * 60)

if __name__ == "__main__":
    main()
