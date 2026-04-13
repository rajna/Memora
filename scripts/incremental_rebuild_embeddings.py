#!/usr/bin/env python3
"""
增量重建 Embedding 脚本
分批次处理，每批 100 个，处理完暂停避免系统终止
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


class SimpleEmbeddingManager:
    """简化版 Embedding 管理器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "embeddings"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._dim = 384
        
    def _get_model(self):
        """懒加载模型"""
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not installed")
            print(f"加载模型: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            print(f"模型加载完成，维度: {self._dim}")
        return self._model
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        model = self._get_model()
        cleaned = text.strip() if text else ""
        embedding = model.encode([cleaned], show_progress_bar=False)
        return embedding[0]
    
    def save_embedding(self, node_id: str, embedding: np.ndarray) -> str:
        """保存向量到文件"""
        file_path = self.cache_dir / f"{node_id}.npy"
        np.save(file_path, embedding)
        return str(file_path)
    
    def load_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """加载向量"""
        file_path = self.cache_dir / f"{node_id}.npy"
        if file_path.exists():
            return np.load(file_path)
        return None


def get_progress_file():
    """获取进度文件路径"""
    return Path(__file__).parent.parent / "embeddings" / ".rebuild_progress.json"


def load_progress():
    """加载进度"""
    progress_file = get_progress_file()
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return json.load(f)
    return {"processed": [], "failed": [], "start_time": datetime.now().isoformat()}


def save_progress(progress):
    """保存进度"""
    progress_file = get_progress_file()
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def log_message(msg):
    """打印并记录日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    
    # 同时写入日志文件
    log_file = Path(__file__).parent.parent / "logs" / "rebuild_embeddings.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(log_line + "\n")


def load_markdown_node(md_path: Path):
    """从 markdown 文件加载节点"""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        
        return {
            "id": post.get("id", md_path.stem),
            "content": post.content,
            "title": post.get("title"),
            "embedding_file": post.get("embedding_file"),
        }
    except Exception as e:
        log_message(f"  加载失败 {md_path}: {e}")
        return None


def save_markdown_node(md_path: Path, node: dict):
    """保存节点回 markdown 文件"""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        
        # 更新 embedding_file
        post.metadata["embedding_file"] = node.get("embedding_file")
        post.metadata["modified"] = datetime.now().isoformat()
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
        
        return True
    except Exception as e:
        log_message(f"  保存失败 {md_path}: {e}")
        return False


def find_missing_embeddings(data_dir, embeddings_dir):
    """
    找出所有缺失 embedding 的节点
    
    Returns:
        List[(node_id, md_file_path)] - 缺失 embedding 的节点列表
    """
    data_path = Path(data_dir)
    embeddings_path = Path(embeddings_dir)
    
    # 获取所有已有的 embedding 文件名（不含扩展名）
    existing_embeddings = set()
    if embeddings_path.exists():
        for f in embeddings_path.glob("*.npy"):
            existing_embeddings.add(f.stem)
    
    log_message(f"已有 embedding 文件: {len(existing_embeddings)} 个")
    
    # 遍历所有 .md 文件
    missing = []
    for md_file in data_path.rglob("*.md"):
        node_id = md_file.stem  # 文件名就是 node_id
        if node_id not in existing_embeddings:
            missing.append((node_id, str(md_file)))
    
    log_message(f"缺失 embedding 的节点: {len(missing)} 个")
    return missing


def process_batch(embed_manager, batch, batch_num, total_batches):
    """
    处理一批节点
    
    Args:
        embed_manager: EmbeddingManager 实例
        batch: [(node_id, md_file_path)] 列表
        batch_num: 当前批次号
        total_batches: 总批次数
    
    Returns:
        (success_count, fail_count, fail_list)
    """
    log_message(f"\n========== 批次 {batch_num}/{total_batches} ==========")
    log_message(f"处理 {len(batch)} 个节点...")
    
    success_count = 0
    fail_count = 0
    fail_list = []
    
    for i, (node_id, md_path) in enumerate(batch, 1):
        try:
            md_file = Path(md_path)
            
            # 加载节点
            node = load_markdown_node(md_file)
            if not node:
                log_message(f"  [{i}/{len(batch)}] ❌ 无法加载: {node_id}")
                fail_count += 1
                fail_list.append((node_id, "load_failed"))
                continue
            
            # 生成 embedding
            content = node.get("content", "")
            if not content.strip():
                log_message(f"  [{i}/{len(batch)}] ⚠ 空内容: {node_id}")
                content = node.get("title", "") or "empty"
            
            embedding = embed_manager.encode_single(content)
            
            # 保存 embedding
            embed_path = embed_manager.save_embedding(node_id, embedding)
            
            # 更新节点的 embedding_file 字段
            node["embedding_file"] = embed_path
            save_markdown_node(md_file, node)
            
            success_count += 1
            
            # 每 10 个显示一次进度
            if i % 10 == 0:
                log_message(f"  [{i}/{len(batch)}] 已处理 {i} 个...")
                
        except Exception as e:
            log_message(f"  [{i}/{len(batch)}] ❌ 错误 {node_id}: {str(e)[:50]}")
            fail_count += 1
            fail_list.append((node_id, str(e)[:100]))
    
    log_message(f"批次完成: ✓ {success_count} 个成功, ✗ {fail_count} 个失败")
    
    return success_count, fail_count, fail_list


def main():
    """主函数"""
    # 路径配置
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    embeddings_dir = base_dir / "embeddings"
    
    log_message("=" * 60)
    log_message("开始增量重建 Embedding")
    log_message("=" * 60)
    
    # 检查 sentence-transformers
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        log_message("❌ 错误: sentence-transformers 未安装")
        log_message("请运行: pip install sentence-transformers")
        sys.exit(1)
    
    # 初始化组件
    log_message("初始化 Embedding 管理器...")
    embed_manager = SimpleEmbeddingManager(cache_dir=str(embeddings_dir))
    
    # 加载进度
    progress = load_progress()
    processed_ids = set(progress.get("processed", []))
    failed_ids = set(progress.get("failed", []))
    
    log_message(f"已处理过的节点: {len(processed_ids)} 个")
    log_message(f"之前失败的节点: {len(failed_ids)} 个")
    
    # 找出所有缺失 embedding 的节点
    missing = find_missing_embeddings(data_dir, embeddings_dir)
    
    # 过滤掉已经处理过的
    missing = [(nid, path) for nid, path in missing if nid not in processed_ids]
    
    if not missing:
        log_message("✓ 所有节点都已有 embedding，无需处理！")
        # 统计最终数量
        final_count = len(list(embeddings_dir.glob("*.npy")))
        log_message(f"Embedding 文件总数: {final_count} 个")
        return
    
    log_message(f"实际需要处理: {len(missing)} 个节点")
    
    # 分批配置
    BATCH_SIZE = 100
    PAUSE_SECONDS = 2  # 每批处理后的暂停时间
    
    total = len(missing)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    
    log_message(f"批次大小: {BATCH_SIZE}, 总批次数: {total_batches}")
    log_message(f"批次间隔: {PAUSE_SECONDS} 秒\n")
    
    # 按批次处理
    total_success = 0
    total_fail = 0
    all_fails = []
    
    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch = missing[start_idx:end_idx]
        
        # 处理当前批次
        success, fail, fails = process_batch(
            embed_manager, batch, batch_num, total_batches
        )
        
        total_success += success
        total_fail += fail
        all_fails.extend(fails)
        
        # 更新进度
        for node_id, _ in batch:
            if node_id not in [f[0] for f in fails]:
                if node_id not in progress["processed"]:
                    progress["processed"].append(node_id)
        for node_id, error in fails:
            if node_id not in progress["failed"]:
                progress["failed"].append(node_id)
        
        save_progress(progress)
        
        # 显示总体进度
        processed_so_far = len(progress["processed"])
        current_npy_count = len(list(embeddings_dir.glob("*.npy")))
        log_message(f"\n总体进度: {processed_so_far} 个已处理, 当前 .npy 文件: {current_npy_count} 个")
        
        # 批次间暂停（最后一批除外）
        if batch_num < total_batches:
            log_message(f"暂停 {PAUSE_SECONDS} 秒...")
            time.sleep(PAUSE_SECONDS)
            
            # 强制垃圾回收，释放内存
            gc.collect()
    
    # 完成总结
    log_message("\n" + "=" * 60)
    log_message("重建完成！")
    log_message("=" * 60)
    log_message(f"本次运行: ✓ {total_success} 个成功, ✗ {total_fail} 个失败")
    
    # 统计最终数量
    final_count = len(list(embeddings_dir.glob("*.npy")))
    log_message(f"Embedding 文件总数: {final_count} 个")
    
    if all_fails:
        log_message(f"\n本次失败列表 (前 10 个):")
        for node_id, error in all_fails[:10]:
            log_message(f"  - {node_id}: {error}")
    
    # 保存最终进度
    progress["end_time"] = datetime.now().isoformat()
    progress["total_success"] = progress.get("total_success", 0) + total_success
    progress["total_fail"] = progress.get("total_fail", 0) + total_fail
    save_progress(progress)
    
    log_message("\n✓ 全部完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("\n⚠ 用户中断，进度已保存")
        sys.exit(1)
    except Exception as e:
        log_message(f"\n❌ 发生错误: {e}")
        import traceback
        log_message(traceback.format_exc())
        sys.exit(1)
