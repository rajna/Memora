# Memora - ai agent记忆系统

<p align="center">
  <img src="logo.png" width="200" alt="Memora Logo">
</p>

> ✨ Memory + Aurora — 记忆极光，让知识自由流动

将 ai agent如nanobot 对话历史转换为"网页"节点，使用 PageRank 算法进行重要性排序，通过多阶段混合检索（TF-IDF + 语义相似度 + PageRank + 图扩散）实现智能检索。

## Core Concept

```
对话回合 → 网页节点 → 向量嵌入 + PageRank分数 + 链接图谱
                ↓
         多阶段混合检索引擎
   (TF-IDF召回 → 子图扩散 → 语义精排)
```

## System Overview

<p align="center">
  <img src="info.png" width="800" alt="Memora System Info">
</p>

## Architecture

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据模型 | `src/models.py` | MemoryNode, SearchResult |
| 存储层 | `src/storage.py` | Markdown + YAML Frontmatter |
| 向量嵌入 | `src/embeddings.py` | sentence-transformers + TF-IDF |
| PageRank | `src/pagerank.py` | NetworkX 图谱算法 |
| 检索引擎 | `src/retrieval.py` | TwoStageRetriever 两阶段检索 |
| 标签生成 | `src/tag_generator.py` | LLM 自动生成标签 |
| 主入口 | `src/memory_system.py` | MemorySystem 统一API |
| Web查看器 | `web/viewer.py` | Flask Web界面 (port 5001) |

## Quick Start

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 初始化系统（首次运行）
python3 tools/init.py

# 3. 启动Web查看器
python web/viewer.py
# 打开 http://localhost:5001

# 4. 运行测试
python tests/test_memory_system.py
```

## Usage

### 基础API

```python
from src.memory_system import MemorySystem

ms = MemorySystem()

# 添加记忆（基础方式）
node = ms.add_memory(
    content="VRM动捕项目使用MediaPipe进行姿态估计...",
    title="VRM Motion Capture",
    tags=["project", "vrm", "motion-capture"]
)

# 从对话消息添加（推荐）
# 自动完成：技能检测 + 格式化 + 标签生成 + 链接建立
node = ms.add_memory_from_messages(
    messages=[
        {"role": "user", "content": "帮我搜索新闻", "timestamp": "2026-04-13T10:00:00"},
        {"role": "assistant", "content": "正在搜索..."},
        {"role": "tool", "content": "name: web-search..."},
    ],
    source="auto-save"  # 标记来源：auto-save, cli-import, manual
)

# 搜索（自动选择最优策略）
results = ms.search("动捕技术", top_k=5)
for r in results:
    print(f"{r.node.title}: {r.final_score:.3f}")

# 查看统计
print(ms.stats())
```

### 高级检索

```python
from src.retrieval import TwoStageRetriever

retriever = TwoStageRetriever(ms.storage)

# Hybrid 检索（统一默认方法）- TF-IDF召回 + 子图扩散 + 语义精排
results = retriever.search("项目bug修复", top_k=5)

# 查看结果来源
for r in results:
    print(f"{r.node.title}: {r.final_score:.3f}")
    if r.metadata.get('is_expanded'):
        print(f"  ↳ 扩散自: {r.metadata['expanded_from']}")

# 基础检索（保留用于对比测试）
results = retriever.search_basic("项目bug修复", top_k=5)
```

### CLI 查询 Skill

```bash
# 安装 skill 后使用
python -m memora_query "那个项目"
python -m memora_query "特朗普" --top-k 10
python -m memora_query "新闻" --tags news,tech --days 7
python -m memora_query --stats  # 查看统计
```

## Retrieval Method (v3.0)

**统一使用 Hybrid 方法** - TF-IDF召回 → 子图扩散 → 语义精排

| 指标 | 数值 |
|------|------|
| **Recall@1** | 80.0% |
| **Recall@5** | 100.0% |
| **MRR** | 0.857 |
| **平均耗时** | ~1.2s/题 |

**流程:**
```
TF-IDF召回10 → 子图扩散(最大50节点) → 语义精排 → top-5
```

**评分公式:**
```
final_score = (0.7 × semantic + 0.3 × pagerank + 0.1 × tfidf) × recency_penalty
```

> 基于 50题 Benchmark 测试 (v3.0) — `search()` 默认调用 Hybrid 方法
> 
> 基础两阶段检索 (`search_basic()`) 保留用于对比测试

## Storage Format

每个记忆节点存储为 Markdown 文件：

```markdown
---
id: 202604081730-a1b2c3d4
url: /memory/2026/04/08/a1b2c3d4
created: 2026-04-08T17:30:00
modified: 2026-04-08T17:30:00
pagerank: 1.2345
source: auto-save        # auto-save | cli-import | manual
tags: [project, vrm, "2026-04-08"]
links:
  - /memory/2026/04/07/xyz789
backlinks:
  - /memory/2026/04/09/abc123
embedding_file: 202604081730-a1b2c3d4.npy
---

[用户] 如何实现VRM动捕？

[AI] VRM动捕项目使用MediaPipe进行姿态估计...
[使用的技能] web-search
```

## Scoring Formula

```
final_score = (0.7 × semantic + 0.3 × pagerank + 0.1 × tfidf) × recency_penalty

where:
- semantic: paraphrase-multilingual-MiniLM-L12-v2 余弦相似度
- pagerank: 基于 links/backlinks 图算法计算（归一化）
- tfidf: jieba + TF-IDF 快速匹配分数
- recency_penalty: 新节点(<7天) 0.3~0.8，成熟节点 1.0
```

## Link Building Strategies

1. **Semantic Similarity** (>0.8): 自动建立语义链接
2. **Temporal Adjacency** (<24h): 时间相近的记忆互相链接
3. **Shared Tags**: 共享标签的记忆建立链接
4. **Manual**: 手动指定 `links` 字段
5. **Incremental**: 新节点自动链接到相似旧记忆

## Auto-Save Hook

nanobot 通过 `on_response` hook 自动保存对话：

```python
# 触发条件
- 对话长度 > 10 字符
- 包含重要关键词 或 长度 > 30 字符

# 处理流程
1. 提取最近3轮对话
2. 检测使用的 skills（从 tool_calls 和对话内容）
3. 格式化对话（添加 [用户]/[AI] 标签）
4. 生成标题和标签
5. 写入 _pending_queue.jsonl
6. Heartbeat 调用 process_queue.py 导入
```

**Skill 检测逻辑**（可靠来源）：
- ✅ `tool_calls.function.name` — 实际调用的函数
- ✅ `user` 输入内容 — 用户意图
- ✅ `assistant` content/reasoning — AI思考过程
- ❌ `tool` 返回内容 — 排除（避免 list_dir 等误报）

## Project Structure

```
Memora/
├── src/                      # 核心源代码
│   ├── config.py             # 配置
│   ├── models.py             # 数据模型 (MemoryNode, SearchResult)
│   ├── storage.py            # 存储层 (Markdown + YAML)
│   ├── embeddings.py         # 向量嵌入（语义+TF-IDF）
│   ├── pagerank.py           # PageRank算法
│   ├── retrieval.py          # 检索引擎（TwoStageRetriever）
│   └── tag_generator.py      # 标签生成
│
├── data/                     # 记忆数据（按日期组织）
│   └── YYYY/MM/DD/
│       └── YYYYMMDD-HHMM-xxxxxxxx.md
│
├── embeddings/               # 向量缓存
│   └── YYYYMMDD-HHMM-xxxxxxxx.npy
│
├── index/                    # 索引文件
│   ├── pagerank.pkl
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_matrix.npz
│
├── tools/                    # 工具脚本
│   ├── init.py               # 系统初始化
│   ├── build_graph.py        # 构建链接图谱
│   ├── build_index.py        # 构建TF-IDF索引
│   ├── generate_tags.py      # 生成标签
│   ├── update_pagerank.py    # 更新PageRank
│   ├── process_queue.py      # 处理待导入队列
│   └── queue_watcher.py      # 队列监控
│
├── benchmark/                # Benchmark测试
│   ├── benchmark_longmemeval.py
│   ├── compare_methods.py
│   └── *.json                # 测试结果
│
├── tests/                    # 单元测试
│   └── test_memory_system.py
│
├── web/                      # Web界面
│   └── viewer.py             # Flask Web查看器
│
├── _pending_queue.jsonl      # 待导入队列
└── README.md
```

## Tools Reference

| 工具 | 用途 | 示例 |
|------|------|------|
| `init.py` | 首次初始化 | `python3 tools/init.py` |
| `build_graph.py` | 构建链接图谱 | `python3 tools/build_graph.py` |
| `build_index.py` | 构建TF-IDF索引 | `python3 tools/build_index.py` |
| `process_queue.py` | 处理待导入队列 | `python3 tools/process_queue.py` |
| `generate_tags.py` | 为所有节点生成标签 | `python3 tools/generate_tags.py` |
| `update_pagerank.py` | 重新计算PageRank | `python3 tools/update_pagerank.py` |
| `debug_graph.py` | 可视化链接图 | `python3 tools/debug_graph.py` |
| `debug_pagerank.py` | 调试PageRank | `python3 tools/debug_pagerank.py` |

## Why Pure Markdown?

- **Human-readable**: 双击即可查看
- **Git-friendly**: diff友好，可版本控制
- **Tool ecosystem**: Obsidian, VSCode 等工具兼容
- **No vendor lock-in**: 纯文件，无数据库依赖
- **Scales well**: 10k+ 文件后才需要优化

## Related Skills

| Skill | 用途 | 位置 |
|-------|------|------|
| **memora-query** | 查询Memora记忆，支持三种检索模式 | `skills/memora-query/` |
| **memora-bridge** | nanobot与Memora桥梁，自动保存对话 | `skills/memora-bridge/` |

## Recent Updates

### 2026-04-13
- ✅ **v3.0 统一 Hybrid 检索** - `search()` 默认使用 Hybrid，简化 API
- ✅ 重构 memora-query skill - 直接调用 Memora 核心，删除独立实现
- ✅ 50题 Benchmark 测试 - Recall@5 100%, MRR 0.857
- ✅ 修复 Skill 检测逻辑（排除 tool 返回内容，避免 list_dir 误报）
- ✅ 统一 `add_memory_from_messages` API（自动检测 skills、格式化、打标签）
- ✅ 优化检索公式权重（0.7语义 + 0.3PageRank + 0.1TF-IDF）
- ✅ 修复时效性惩罚（新节点降权，成熟节点1.0）

### 2026-04-12
- ✅ TwoStageRetriever 三阶段检索（TF-IDF召回 + 子图扩散 + 语义精排）
- ✅ 三种检索方法完成基准测试（hybrid 86.7% Recall@5）
- ✅ 增量链接机制（新节点自动链接相似旧记忆）
- ✅ auto-save hook v6 简化版（只检测存在的 skill）

## License

MIT
