#!/usr/bin/env python3
"""
TextRank Tags 生成器 - 便捷入口
使用方法:
    python tools/generate_tags.py                  # 生成标签并保存
    python tools/generate_tags.py --apply          # 生成并应用到记忆节点
    python tools/generate_tags.py --features 500   # 设置特征数
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入模块（使用 -m 方式运行时相对导入会正常工作）
from src.tag_generator import TagGenerator, TagGenerationResult
from src.storage import MemoryStorage
from src import config

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TextRank 标签生成器')
    parser.add_argument('--features', '-f', type=int, default=1000,
                        help='TF-IDF 最大特征数 (默认: 1000)')
    parser.add_argument('--output', '-o', type=str, 
                        default=os.path.join(config.INDEX_DIR, 'tags_generated.json'),
                        help='输出文件路径')
    parser.add_argument('--apply', '-a', action='store_true',
                        help='将标签应用到记忆节点')
    
    args = parser.parse_args()
    
    # 检查依赖
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("❌ 需要安装 scikit-learn: pip install scikit-learn")
        return 1
    
    # 生成标签
    generator = TagGenerator()
    result = generator.generate_tags(max_features=args.features)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.save(args.output)
    print(f"💾 结果已保存: {args.output}")
    
    # 应用标签
    if args.apply:
        generator.apply_tags_to_nodes(result)
    
    # 打印摘要
    print("\n" + "="*50)
    print("📊 标签生成摘要")
    print("="*50)
    print(f"总记忆节点: {result.total_nodes}")
    print(f"生成标签数: {len(result.node_tags)}")
    print(f"\n🔥 全局关键词 Top 10:")
    for word, score in result.global_keywords[:10]:
        print(f"  • {word}: {score:.4f}")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    exit(main())
