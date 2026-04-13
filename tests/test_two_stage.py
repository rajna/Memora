#!/usr/bin/env python3
"""
测试两阶段检索性能
对比：标准检索 vs 两阶段检索
"""
import sys
import time
sys.path.insert(0, '.')

# 绕过依赖检查，直接测试核心逻辑
exec(open('src/embeddings.py').read())

print("=" * 60)
print("两阶段检索性能测试")
print("=" * 60)

# 创建测试数据
print("\n1. 创建测试数据...")
test_texts = [
    "VRM动捕项目使用MediaPipe进行姿态估计",
    "Python编程语言的基础教程",
    "机器学习是人工智能的核心技术",
    "深度学习在图像识别中的应用",
    "自然语言处理技术进展",
    "Transformer模型架构详解",
    "PyTorch深度学习框架入门",
    "计算机视觉中的目标检测",
    "强化学习在游戏AI中的应用",
    "神经网络优化算法综述",
    "生成对抗网络GAN原理",
    "BERT预训练模型解读",
    "GPT大语言模型技术",
    "注意力机制Attention详解",
    "卷积神经网络CNN结构",
    "循环神经网络RNN与LSTM",
    "图神经网络GNN应用",
    "多模态学习最新进展",
    "自监督学习方法对比",
    "模型压缩与量化技术",
]

print(f"   创建了 {len(test_texts)} 条测试文本")

# 测试 sentence-transformers
print("\n2. 测试 sentence-transformers 性能...")
try:
    semantic_mgr = EmbeddingManager(model_name='sentence-transformers/all-MiniLM-L6-v2')
    print("   ✅ 模型加载成功")
    
    # 批量编码
    start = time.time()
    semantic_vecs = semantic_mgr.encode(test_texts)
    semantic_time = time.time() - start
    print(f"   ✅ 批量编码 {len(test_texts)} 条: {semantic_time:.3f}s")
    print(f"   向量维度: {semantic_vecs.shape}")
    
    # 单条编码（模拟查询）
    start = time.time()
    for _ in range(10):
        _ = semantic_mgr.encode_single("机器学习")
    single_time = (time.time() - start) / 10
    print(f"   单条编码平均: {single_time:.3f}s")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    semantic_mgr = None

# 测试 TF-IDF
print("\n3. 测试 TF-IDF + jieba 性能...")
tfidf_mgr = TFIDFEmbeddingManager(max_features=1000)

start = time.time()
tfidf_mgr.fit(test_texts)
fit_time = time.time() - start
print(f"   ✅ Fit 时间: {fit_time:.3f}s")

start = time.time()
tfidf_vecs = tfidf_mgr.encode(test_texts)
tfidf_time = time.time() - start
print(f"   ✅ 批量编码 {len(test_texts)} 条: {tfidf_time:.3f}s")
print(f"   向量维度: {tfidf_vecs.shape}")

start = time.time()
for _ in range(100):
    _ = tfidf_mgr.encode_single("机器学习")
single_tfidf_time = (time.time() - start) / 100
print(f"   单条编码平均: {single_tfidf_time:.3f}s (比 semantic 快 {single_time/single_tfidf_time:.1f}x)")

# 相似度质量对比
print("\n4. 相似度质量对比...")

queries = [
    "机器学习算法",
    "深度学习模型",
    "神经网络架构",
    "Python编程",
]

for query in queries:
    print(f"\n   查询: '{query}'")
    
    # TF-IDF 相似度
    q_tfidf = tfidf_mgr.encode_single(query)
    tfidf_sims = []
    for i, vec in enumerate(tfidf_vecs):
        norm_q = np.linalg.norm(q_tfidf)
        norm_v = np.linalg.norm(vec)
        if norm_q > 0 and norm_v > 0:
            sim = np.dot(q_tfidf, vec) / (norm_q * norm_v)
            sim = (sim + 1) / 2
        else:
            sim = 0
        tfidf_sims.append((i, sim))
    tfidf_sims.sort(key=lambda x: x[1], reverse=True)
    
    # Semantic 相似度
    if semantic_mgr:
        q_sem = semantic_mgr.encode_single(query)
        sem_sims = []
        for i, vec in enumerate(semantic_vecs):
            norm_q = np.linalg.norm(q_sem)
            norm_v = np.linalg.norm(vec)
            if norm_q > 0 and norm_v > 0:
                sim = np.dot(q_sem, vec) / (norm_q * norm_v)
                sim = (sim + 1) / 2
            else:
                sim = 0
            sem_sims.append((i, sim))
        sem_sims.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   TF-IDF Top 3: {[test_texts[i][:20] + '...' for i, _ in tfidf_sims[:3]]}")
    if semantic_mgr:
        print(f"   Semantic Top 3: {[test_texts[i][:20] + '...' for i, _ in sem_sims[:3]]}")

# 两阶段检索模拟
print("\n5. 模拟两阶段检索效率...")
if semantic_mgr:
    # 模拟 1000 条数据的场景
    n_total = 1000
    n_candidates = 100
    
    # 方法1: 纯语义检索（编码所有）
    pure_semantic_time = n_total * single_time
    
    # 方法2: 两阶段检索
    # 阶段1: TF-IDF 筛选
    stage1_time = n_total * single_tfidf_time
    # 阶段2: 语义精排 top-100
    stage2_time = n_candidates * single_time
    two_stage_time = stage1_time + stage2_time
    
    speedup = pure_semantic_time / two_stage_time
    
    print(f"   假设总数据量: {n_total} 条")
    print(f"   纯语义检索: {pure_semantic_time:.3f}s")
    print(f"   两阶段检索: {two_stage_time:.3f}s (TF-IDF: {stage1_time:.3f}s + Semantic: {stage2_time:.3f}s)")
    print(f"   ⚡ 加速比: {speedup:.1f}x")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
