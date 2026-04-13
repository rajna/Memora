# -*- coding: utf-8 -*-
"""
Web Viewer for Memory System
记忆系统Web查看器

Usage:
    cd /Users/rama/.nanobot/workspace/Memora
    python web/viewer.py
    
Then open http://localhost:5000
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, render_template_string, request, jsonify

# 修复相对导入问题
import src.models as models
import src.config as config
sys.modules['models'] = models
sys.modules['config'] = config

from src.memory_system import MemorySystem

app = Flask(__name__)
ms = MemorySystem()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Memory System - 网页记忆</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .search-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .search-box input {
            width: 70%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .search-box button {
            padding: 10px 20px;
            font-size: 16px;
            background: #333;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .results {
            display: grid;
            gap: 15px;
        }
        .memory-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .memory-card h3 {
            margin: 0;
            color: #333;
        }
        .memory-card .url {
            color: #666;
            font-size: 12px;
            margin-top: 5px;
        }
        .memory-card .content {
            margin-top: 10px;
            color: #444;
            line-height: 1.6;
        }
        .memory-card .tags {
            margin-top: 10px;
        }
        .memory-card .tag {
            display: inline-block;
            background: #e0e0e0;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin-right: 5px;
        }
        .memory-card .score {
            float: right;
            color: #4CAF50;
            font-weight: bold;
        }
        .stats {
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>🔍 Memory System - 网页记忆</h1>
    
    <div class="stats">
        {{ stats.total_nodes }} memories | 
        {{ stats.total_tags }} tags | 
        Avg PageRank: {{ "%.4f"|format(stats.avg_pagerank) }}
    </div>
    
    <div class="search-box">
        <form method="GET" action="/">
            <input type="text" name="q" value="{{ query }}" placeholder="搜索记忆...">
            <button type="submit">搜索</button>
        </form>
    </div>
    
    <div class="results">
        {% for result in results %}
        <div class="memory-card">
            <span class="score">{{ "%.3f"|format(result.final_score) }}</span>
            <h3>{{ result.node.title or "Untitled" }}</h3>
            <div class="url">{{ result.node.url }}</div>
            <div class="content">{{ result.node.content[:300] }}{% if result.node.content|length > 300 %}...{% endif %}</div>
            <div class="tags">
                {% for tag in result.node.tags %}
                <span class="tag">{{ tag }}</span>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    query = request.args.get('q', '')
    # Fix URL encoding for Chinese characters
    if query:
        try:
            # Try to fix mojibake (UTF-8 decoded as Latin-1)
            query = query.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    
    if query:
        results = ms.search(query, top_k=20)
    else:
        # 显示最近添加的记忆
        nodes = ms.list_all(limit=20)
        results = []
        for node in nodes:
            from models import SearchResult
            results.append(SearchResult(
                node=node,
                semantic_score=0,
                pagerank_score=node.pagerank,
                recency_score=1.0,
                final_score=node.pagerank
            ))
    
    stats = ms.stats()
    
    return render_template_string(
        HTML_TEMPLATE,
        results=results,
        query=query,
        stats=stats
    )


@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')
    # Fix URL encoding for Chinese characters
    if query:
        try:
            query = query.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    top_k = int(request.args.get('k', 10))
    
    results = ms.search(query, top_k=top_k)
    
    return jsonify({
        "query": query,
        "results": [r.to_dict() for r in results]
    })


@app.route('/api/memory/<path:url>')
def api_memory(url):
    node_path = "/memory/" + url
    node = ms.get(node_path)
    if node:
        return jsonify(node.to_dict())
    return jsonify({"error": "Not found"}), 404


if __name__ == '__main__':
    print("Starting Memory System Web Viewer...")
    port = int(os.environ.get('PORT', 5555))  # 默认5555
    print(f"Open http://localhost:{port} in your browser")
    app.run(host='0.0.0.0', port=port, debug=False)
