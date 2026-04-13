# TF-IDF Tags 生成器

自动为记忆节点生成标签，使用 TF-IDF 提取关键词。

## 功能

- **TF-IDF 关键词提取**: 识别每个记忆节点的重要内容
- **智能标签**: 每个节点独立提取权重最高的词作为标签
- **中文支持**: 使用 jieba 进行中文分词

## 使用方法

### 基本用法

```bash
# 进入 Memora 目录
cd /Users/rama/.nanobot/workspace/Memora

# 运行标签生成器
python3 tools/generate_tags.py

# 生成并应用标签到记忆节点
python3 tools/generate_tags.py --apply
```

### 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--features` | `-f` | TF-IDF 最大特征数 | 1000 |
| `--apply` | `-a` | 将标签应用到记忆节点 | False |
| `--output` | `-o` | 输出文件路径 | index/tags_generated.json |

### 示例

```bash
# 使用更多特征
python3 tools/generate_tags.py --features 2000

# 生成并应用标签到记忆节点
python3 tools/generate_tags.py --apply

# 指定输出路径
python3 tools/generate_tags.py -o my_tags.json
```

## 输出格式

生成的 JSON 文件包含：

```json
{
  "node_tags": {
    "记忆ID": ["标签1", "标签2", "标签3"],
    ...
  },
  "global_keywords": [
    ["关键词", 0.0549],
    ...
  ],
  "generated_at": "2026-04-10T09:00:00",
  "total_nodes": 244
}
```

## 算法说明

### 1. 文本预处理
- 移除代码块、URL、特殊字符
- 中文使用 jieba 分词
- 过滤停用词和单字

### 2. TF-IDF 计算
- 使用 1-2 gram
- 最小文档频率: 2
- 最大文档频率: 80%

### 3. 标签生成
- 每个节点取 TF-IDF 权重最高的词作为标签
- 默认每个节点 3-5 个标签

## 依赖安装

```bash
pip install scikit-learn jieba
```

## 在代码中使用

```python
from src.tag_generator import TagGenerator

# 创建生成器
generator = TagGenerator()

# 生成标签
result = generator.generate_tags(
    max_features=1000,   # TF-IDF特征数
    top_k=3              # 每个节点标签数
)

# 应用标签到记忆节点
generator.apply_tags_to_nodes(result)

# 保存结果
result.save('tags.json')
```

## 注意事项

1. **单节点支持**: 即使只有 1 个节点也能提取标签
2. **内存占用**: 大量节点时可能需要调整 `max_features`
3. **中文分词**: 首次运行需要加载 jieba 词典，约 1-2 秒
