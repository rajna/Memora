#!/usr/bin/env python3
"""测试 TF-IDF + jieba 嵌入管理器"""
import sys
sys.path.insert(0, '.')

# 直接导入 embeddings 模块
exec(open('src/embeddings.py').read())

print("=" * 50)
print("测试 TF-IDF + jieba 嵌入管理器")
print("=" * 50)

# 测试 TF-IDF 版本
print("\n1. 创建 TFIDFEmbeddingManager...")
tfidf_mgr = TFIDFEmbeddingManager(max_features=100)
print(f"   ✅ 创建成功 (jieba可用: {tfidf_mgr._use_jieba})")

# 测试编码
print("\n2. 测试编码...")
test_texts = [
    '这是一个测试',
    'Python programming language',
    '机器学习是人工智能的重要分支',
    '深度学习在图像识别中的应用',
    '自然语言处理技术'
]
vectors = tfidf_mgr.encode(test_texts)
print(f"   ✅ 编码成功: {vectors.shape}")

# 测试相似度计算
print("\n3. 测试相似度计算...")
sim_matrix = []
for i in range(len(test_texts)):
    row = []
    for j in range(len(test_texts)):
        sim = tfidf_mgr.compute_similarity(
            vectors[i], vectors[j],
            test_texts[i], test_texts[j]
        )
        row.append(sim)
    sim_matrix.append(row)

print("   相似度矩阵:")
for i, row in enumerate(sim_matrix):
    print(f"   文本{i}: {[f'{s:.2f}' for s in row]}")

# 测试 jieba 分词
print("\n4. 测试 jieba 分词...")
if tfidf_mgr._use_jieba:
    tokens = tfidf_mgr._tokenize('自然语言处理是人工智能的重要分支')
    print(f"   分词结果: {tokens}")
else:
    print("   ⚠️ jieba 未安装")

# 测试工厂函数
print("\n5. 测试工厂函数...")
for backend in ['tfidf', 'simple']:
    mgr = get_embedding_manager(backend)
    print(f"   {backend}: {type(mgr).__name__}")

print("\n" + "=" * 50)
print("所有测试通过! ✅")
print("=" * 50)
