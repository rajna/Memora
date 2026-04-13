"""
TextRank Tags Generator (No Clustering)
TextRank 标签生成器（无聚类版）

为每个记忆节点独立生成标签：
1. 提取节点内容
2. TextRank 图算法提取关键词（单文档内模拟 IDF 效果）
3. 备选：TF-IDF / 词频统计
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 检查 sklearn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn 未安装，运行: pip install scikit-learn")

# jieba 用于中文分词和关键词提取
try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("⚠️  jieba 未安装，运行: pip install jieba")

from .storage import MemoryStorage
from .models import MemoryNode
from . import config


@dataclass
class TagGenerationResult:
    """标签生成结果"""
    node_tags: Dict[str, List[str]]      # 每个记忆节点的标签
    global_keywords: List[Tuple[str, float]]  # 全局关键词及权重
    generated_at: datetime
    total_nodes: int
    
    def to_dict(self) -> Dict:
        return {
            "node_tags": self.node_tags,
            "global_keywords": self.global_keywords,
            "generated_at": self.generated_at.isoformat(),
            "total_nodes": self.total_nodes,
        }
    
    def save(self, filepath: str):
        """保存结果到 JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class TagGenerator:
    """
    TextRank 标签生成器 - 无聚类版本
    每个节点独立提取标签，优先使用 TextRank 图算法
    """
    
    # 停用词列表
    STOPWORDS = {
        # 中文停用词
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', 
        '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
        '之', '与', '及', '等', '或', '但', '而', '为', '于', '以', '被', '将', '把', '让', '使',
        '这个', '那个', '这些', '那些', '这里', '那里', '这样', '那样', '什么', '怎么', '如何',
        '可以', '需要', '进行', '通过', '根据', '关于', '对于', '如果', '因为', '所以', '虽然',
        '用户', '问题', '回答', '帮助', '请问', '谢谢', '您好',
        # 英文停用词 - 完整列表
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
        'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'am', 's', 't', 'don', 'doesn', 'didn', 'wasn', 'weren',
        'haven', 'hasn', 'hadn', 'won', 'wouldn', 'couldn', 'shouldn', 'aren', 'isn', 'let',
        'm', 're', 've', 'd', 'll', 'y', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
        'wasn', 'weren', 'won', 'wouldn', 'yourselves', 'whom',
        # 数据集/对话通用词
        'session', 'sessions', 'user', 'assistant', 'dialogue', 'dialogues', 'conversation',
        'conversations', 'question', 'questions', 'answer', 'answers', 'message', 'messages',
        'content', 'context', 'role', 'text', 'data'
    }
    
    def __init__(self, storage: Optional[MemoryStorage] = None):
        if storage is None:
            self.storage = MemoryStorage(config.MEMORY_DIR)
        else:
            self.storage = storage
        
        self.nodes: List[MemoryNode] = []
        
    def _is_node_id(self, text: str) -> bool:
        """检查文本是否像节点ID（需要过滤）"""
        text = text.strip().lower()
        
        # lme- 开头的（LongMemEval ID）
        if text.startswith('lme-'):
            return True
        
        # 8位十六进制（如 94f70d80）
        if len(text) == 8 and all(c in '0123456789abcdef' for c in text):
            return True
        
        # 纯数字（可能是日期/时间戳）
        if text.isdigit() and len(text) >= 4:
            return True
        
        # session_id 格式（如 0104, 0928）
        if len(text) == 4 and text.isdigit():
            return True
            
        return False
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""
        
        # 移除代码块、URL、特殊字符
        text = re.sub(r'```[\s\S]*?```', ' ', text)
        text = re.sub(r'`[^`]*`', ' ', text)
        text = re.sub(r'https?://\S+', ' ', text)
        
        # 移除看起来像节点ID的词
        text = re.sub(r'lme-[a-f0-9]{8}-\d{4}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[a-f0-9]{8}\b', ' ', text, flags=re.IGNORECASE)
        
        # 保留中英文和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        
        return text.strip()
    
    def _extract_tags_from_node(self, node: MemoryNode, top_k: int = 15) -> List[str]:
        """
        从单个节点提取标签
        
        优先使用 jieba 的 textrank 提取关键词
        如果不支持，则使用简单的词频统计
        """
        # 合并标题和内容，限制长度避免处理超大文件
        # 只取前3000字符，通常足以代表主题
        content_preview = node.content[:3000] if node.content else ""
        full_text = f"{node.title or ''} {content_preview}"
        
        if not full_text.strip():
            return []
        
        # 清理文本
        clean_text = self._preprocess_text(full_text)
        
        tags = []
        
        # 方法1: 使用 jieba 的 textrank 提取关键词（更精准）
        # TextRank: 基于词共现图算法，在单文档内模拟 IDF 效果
        if JIEBA_AVAILABLE:
            try:
                # textrank 提取关键词
                # allowPOS: 名词、动词、动名词、专有名词、英文
                keywords = jieba.analyse.textrank(
                    clean_text, 
                    topK=top_k * 2,  # 多提取一些用于过滤
                    withWeight=True,
                    allowPOS=('n', 'v', 'vn', 'nz', 'eng')
                )
                
                for word, weight in keywords:
                    word = word.strip().lower()
                    # 过滤停用词、节点ID和太短的词
                    if (len(word) >= 2 and 
                        word not in self.STOPWORDS and
                        not word.isdigit() and
                        not self._is_node_id(word) and
                        word not in tags):
                        tags.append(word)
                    
                    if len(tags) >= top_k:
                        break
                        
            except Exception as e:
                pass  # 失败则使用备选方法
        
        # 方法2: 简单的 TF-IDF 提取（备选）
        if len(tags) < top_k and SKLEARN_AVAILABLE:
            try:
                # 将文本分成句子/片段
                sentences = re.split(r'[。\.\n]+', clean_text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                
                if len(sentences) >= 2:
                    # 使用 TF-IDF
                    vectorizer = TfidfVectorizer(
                        max_features=100,
                        ngram_range=(1, 2),
                        stop_words=list(self.STOPWORDS),
                        min_df=1,
                        max_df=1.0
                    )
                    
                    tfidf_matrix = vectorizer.fit_transform(sentences)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # 计算每个词的平均权重
                    mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
                    top_indices = mean_scores.argsort()[-(top_k * 2):][::-1]
                    
                    for idx in top_indices:
                        word = feature_names[idx]
                        if (mean_scores[idx] > 0 and 
                            len(word) >= 2 and
                            word not in self.STOPWORDS and
                            not word.isdigit() and
                            not self._is_node_id(word) and
                            word not in tags):
                            tags.append(word)
                        
                        if len(tags) >= top_k:
                            break
                            
            except Exception as e:
                pass
        
        # 方法3: 如果上述都失败，使用简单词频
        if len(tags) < 3:
            try:
                if JIEBA_AVAILABLE:
                    words = jieba.lcut(clean_text)
                else:
                    words = clean_text.lower().split()
                
                # 统计词频
                from collections import Counter
                word_counts = Counter(
                    w.strip().lower() for w in words 
                    if len(w.strip()) >= 2 
                    and w.strip().lower() not in self.STOPWORDS
                    and not w.strip().isdigit()
                )
                
                for word, count in word_counts.most_common(top_k * 3):
                    if (word not in tags and 
                        not self._is_node_id(word)):
                        tags.append(word)
                    if len(tags) >= top_k:
                        break
                        
            except Exception as e:
                pass
        
        return tags[:top_k]
    
    def generate_tags(self, top_k: int = 5, target_dir: Optional[Path] = None) -> TagGenerationResult:
        """
        生成标签的主函数
        
        Args:
            top_k: 每个节点的标签数
            target_dir: 指定目录，None则处理所有节点
        """
        print(f"🏷️ 开始生成标签 (top_k={top_k})...")
        
        node_tags = {}
        all_keywords = []
        
        # 加载节点
        if target_dir:
            print(f"📁 处理目录: {target_dir}")
            nodes = []
            for file_path in target_dir.rglob("*.md"):
                node = self.storage.load(str(file_path))
                if node:
                    nodes.append(node)
        else:
            print(f"📚 加载所有记忆节点...")
            nodes = list(self.storage.iterate_all())
        
        self.nodes = nodes
        
        print(f"✅ 加载了 {len(nodes)} 个记忆节点")
        
        if len(nodes) == 0:
            return TagGenerationResult(
                node_tags={},
                global_keywords=[],
                generated_at=datetime.now(),
                total_nodes=0
            )
        
        # 为每个节点独立提取标签
        for i, node in enumerate(nodes):
            if i % 50 == 0:
                print(f"  处理进度: {i}/{len(nodes)}")
            
            tags = self._extract_tags_from_node(node, top_k=top_k)
            node_tags[node.id] = tags
            all_keywords.extend(tags)
        
        # 统计全局关键词
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        global_keywords = [(word, count) for word, count in keyword_counts.most_common(50)]
        
        result = TagGenerationResult(
            node_tags=node_tags,
            global_keywords=global_keywords,
            generated_at=datetime.now(),
            total_nodes=len(nodes)
        )
        
        print(f"✅ 标签生成完成！共 {len(node_tags)} 个节点")
        return result
    
    def apply_tags_to_nodes(self, result: TagGenerationResult, save: bool = True, 
                            replace: bool = True, target_dir: Optional[Path] = None):
        """将生成的标签应用到记忆节点
        
        Args:
            result: 标签生成结果
            save: 是否保存到文件
            replace: 是否完全替换旧标签（而非合并）
            target_dir: 指定目录（用于非标准ID格式的节点）
        """
        print(f"💾 应用标签到记忆节点...")
        
        import frontmatter
        
        updated_count = 0
        for node_id, tags in result.node_tags.items():
            # 尝试直接通过文件路径加载（适用于非标准ID格式如 lme-xxx）
            if target_dir:
                file_path = target_dir / f"{node_id}.md"
                if file_path.exists():
                    post = frontmatter.load(file_path)
                    old_tags = post.get('tags', [])
                    
                    if tags != old_tags:
                        post.metadata['tags'] = tags
                        if save:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(frontmatter.dumps(post))
                        updated_count += 1
                    continue
            
            # 使用 storage 加载（标准ID格式）
            node = self.storage.load_by_id(node_id)
            if node:
                if replace:
                    new_tags = tags
                else:
                    existing_tags = set(node.tags or [])
                    preserved = {t for t in existing_tags if not t.startswith('auto-')}
                    new_tags = list(preserved | set(tags))
                
                if new_tags != node.tags:
                    node.tags = new_tags
                    if save:
                        self.storage.save(node)
                    updated_count += 1
        
        print(f"✅ 更新了 {updated_count} 个记忆节点的标签")
        return updated_count


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TextRank 标签生成器（无聚类）')
    parser.add_argument('--dir', '-d', type=str, default=None,
                        help='指定目录生成标签')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='每个节点的标签数 (默认: 5)')
    parser.add_argument('--output', '-o', type=str, 
                        default=os.path.join(config.INDEX_DIR, 'tags_generated.json'),
                        help='输出文件路径')
    parser.add_argument('--apply', '-a', action='store_true',
                        help='将标签应用到记忆节点')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示标签，不保存')
    
    args = parser.parse_args()
    
    # 生成标签
    generator = TagGenerator()
    
    target_dir = Path(args.dir) if args.dir else None
    result = generator.generate_tags(top_k=args.top_k, target_dir=target_dir)
    
    # 保存结果
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        result.save(args.output)
        print(f"💾 结果已保存: {args.output}")
    
    # 应用标签
    if args.apply and not args.dry_run:
        generator.apply_tags_to_nodes(result)
    
    # 打印摘要
    print("\n" + "="*50)
    print("📊 标签生成摘要")
    print("="*50)
    print(f"总记忆节点: {result.total_nodes}")
    print(f"\n🔥 全局高频标签 Top 20:")
    for word, count in result.global_keywords[:20]:
        bar = '█' * min(count, 20)
        print(f"  • {word:20s}: {count:3d} {bar}")
    
    print(f"\n🏷️ 节点标签示例 (前5个):")
    for i, (node_id, tags) in enumerate(list(result.node_tags.items())[:5]):
        print(f"  {node_id[:30]:30s}: {tags}")
    
    print("="*50)
    
    return 0


if __name__ == "__main__":
    exit(main())
