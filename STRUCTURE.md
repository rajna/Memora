# Memora 目录结构

```
Memora/
├── src/              # 核心源代码
│   ├── config.py           # 配置
│   ├── models.py           # 数据模型
│   ├── storage.py          # 存储层
│   ├── embeddings.py       # 向量嵌入
│   ├── pagerank.py         # PageRank算法
│   ├── retrieval.py        # 检索引擎
│   └── tag_generator.py    # 标签生成
│
├── data/             # 记忆数据（按日期组织）
│   └── YYYY/MM/DD/
│       └── YYYYMMDD-HHMM-xxxxxxxx.json
│
├── embeddings/       # 向量缓存
│   └── YYYYMMDD-HHMM-xxxxxxxx.npy
│
├── index/            # 索引文件
│   ├── pagerank.pkl
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_matrix.npz
│
├── tools/            # 工具脚本
│   ├── init.py             # 系统初始化
│   ├── build_graph.py      # 构建链接图
│   ├── build_index.py      # 构建TF-IDF索引
│   ├── generate_tags.py    # 生成标签
│   ├── update_pagerank.py  # 更新PageRank
│   ├── process_queue.py    # 处理队列
│   ├── queue_watcher.py    # 队列监控
│   ├── debug_graph.py      # 调试图
│   ├── debug_pagerank.py   # 调试PR
│   └── ...
│
├── tests/            # 单元测试
│   ├── test_tfidf.py
│   ├── test_two_stage.py
│   ├── test_recency_fix.py
│   └── ...
│
├── benchmark/        # Benchmark测试
│   ├── benchmark.json              # 测试集定义
│   ├── run_conceptual_benchmark.py # 运行测试
│   └── conceptual_result_*.json    # 测试结果
│
├── docs/             # 文档
├── web/              # Web界面
├── scripts/          # 其他脚本
├── README.md
├── BENCHMARK_REPORT.md
└── requirements.txt
```

## 快速使用

```bash
# 初始化系统
python3 tools/init.py

# 生成标签
python3 tools/generate_tags.py --all

# 运行Benchmark
cd benchmark && python3 run_conceptual_benchmark.py

# 调试
python3 tools/debug_pagerank.py
python3 tools/debug_graph.py
```
