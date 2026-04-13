#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 PageRank 图结构并可视化
Usage: python tools/generate_pagerank_graph.py [--output graph.html]
"""
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import MemoryStorage
from src.models import MemoryNode
from src.pagerank import MemoryGraph


def load_all_nodes(data_dir: str) -> List[MemoryNode]:
    """加载所有记忆节点（跳过损坏的文件）"""
    storage = MemoryStorage(data_dir)
    nodes = []
    errors = 0
    
    for file_path in storage.base_dir.rglob("*.md"):
        try:
            node = storage.load(str(file_path))
            if node:
                nodes.append(node)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"   [WARN] Skip damaged file: {file_path.name}")
    
    if errors > 0:
        print(f"   [WARN] Total damaged files: {errors}")
    
    return nodes


def build_graph_stats(nodes: List[MemoryNode]) -> Dict[str, Any]:
    """构建图统计信息"""
    # 按来源目录统计
    source_dirs = Counter()
    for node in nodes:
        if hasattr(node, 'url') and node.url:
            # 从URL提取日期目录
            parts = node.url.split('/')
            if len(parts) >= 4:
                source_dirs[f"{parts[2]}/{parts[3]}"] += 1
        # 检查 embedding_file 路径
        if hasattr(node, 'embedding_file') and node.embedding_file:
            parts = node.embedding_file.split('/')
            if 'embeddings' in parts:
                idx = parts.index('embeddings')
                if idx > 0:
                    source_dirs[parts[idx-1]] += 1
    
    return {
        'total_nodes': len(nodes),
        'nodes_with_links': sum(1 for n in nodes if n.links),
        'nodes_with_backlinks': sum(1 for n in nodes if n.backlinks),
        'total_edges': sum(len(n.links) for n in nodes),
        'source_distribution': dict(source_dirs.most_common(10)),
    }


def generate_graph_data(nodes: List[MemoryNode], max_nodes: int = 1000, top_k: int = 3) -> Dict[str, Any]:
    """
    生成图可视化数据
    使用 auto_build_links(top_k=3) 三重链接策略
    """
    graph = MemoryGraph()
    graph.auto_build_links(nodes, top_k=top_k)
    
    # 计算 PageRank
    pagerank_scores = graph.calculate_pagerank()
    
    # 获取前N个节点
    top_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
    top_node_urls = {url for url, _ in top_nodes}
    
    # 构建节点数据
    node_data = []
    for url, score in top_nodes:
        node = graph.nodes.get(url)
        if not node:
            continue
        
        node_data.append({
            'id': node.url,
            'label': node.title[:30] if node.title else node.id[:8],
            'score': round(score, 6),
            'links_count': len(node.links),
            'backlinks_count': len(node.backlinks),
            'in_degree': len(graph.get_backlinks(url)),
            'out_degree': len(graph.get_outgoing_links(url)),
        })
    
    # 构建边数据 - 从NetworkX图中读取边类型
    edges = []
    edge_type_count = {"semantic": 0, "temporal": 0, "tag": 0, "default": 0}
    
    for url, score in top_nodes:
        node = graph.nodes.get(url)
        if not node:
            continue
        
        # 从NetworkX图中获取出边及其类型
        if url in graph.graph:
            for target_url in graph.graph.successors(url):
                if target_url in top_node_urls:  # 只保留目标也在top N的边
                    edge_data = graph.graph.get_edge_data(url, target_url)
                    edge_type = edge_data.get("edge_type", "default") if edge_data else "default"
                    edge_type_count[edge_type] = edge_type_count.get(edge_type, 0) + 1
                    
                    edges.append({
                        'source': url,
                        'target': target_url,
                        'weight': 1.0,
                        'edge_type': edge_type,
                    })
    
    return {
        'nodes': node_data,
        'edges': edges,
        'stats': {
            'total_nodes': len(nodes),
            'graph_nodes': len(top_nodes),
            'graph_edges': len(edges),
        },
        'edge_type_count': edge_type_count,
    }


def generate_html_visualization(graph_data: dict, output_path: str):
    """生成交互式HTML可视化"""
    import json
    import os
    
    # 获取 logo 路径
    script_dir = Path(__file__).parent.parent
    logo_path = script_dir / "logo.png"
    logo_base64 = ""
    if logo_path.exists():
        import base64
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Memora - Agent Memory</title>
    <link rel="icon" href="data:image/png;base64,{logo_base64}" type="image/png">
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            margin: 0;
            padding: 30px;
            min-height: 100vh;
            background: 
                radial-gradient(ellipse at 50% 0%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 50%, rgba(138, 43, 226, 0.05) 0%, transparent 40%),
                radial-gradient(ellipse at 20% 80%, rgba(0, 255, 136, 0.04) 0%, transparent 40%),
                linear-gradient(180deg, #0a0e17 0%, #0d1117 50%, #0a0e17 100%);
            background-attachment: fixed;
            color: #c9d1d9;
            position: relative;
            overflow-x: hidden;
        }}
        /* Cyber grid overlay */
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }}
        /* Scanline effect */
        body::after {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0, 0, 0, 0.1) 2px,
                rgba(0, 0, 0, 0.1) 4px
            );
            pointer-events: none;
            z-index: 1;
            opacity: 0.3;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            position: relative;
            z-index: 2;
        }}
        .header {{
            display: flex;
            align-items: center;
            gap: 25px;
            margin-bottom: 30px;
            padding: 20px 30px;
            background: linear-gradient(135deg, rgba(13, 17, 23, 0.95) 0%, rgba(22, 27, 34, 0.9) 100%);
            border-radius: 12px;
            border: 1px solid rgba(0, 212, 255, 0.2);
            box-shadow: 
                0 0 20px rgba(0, 212, 255, 0.1),
                0 8px 32px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
            position: relative;
            overflow: hidden;
        }}
        /* Glowing edge effect */
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.5), transparent);
        }}
        .header::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.3), transparent);
        }}
        .logo {{
            width: 55px;
            height: 55px;
            border-radius: 10px;
            box-shadow: 
                0 0 25px rgba(0, 212, 255, 0.3),
                0 4px 15px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(0, 212, 255, 0.3);
        }}
        h1 {{ 
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: #00d4ff; 
            margin: 0;
            font-size: 26px;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
            letter-spacing: 3px;
        }}
        .subtitle {{
            font-family: 'JetBrains Mono', monospace;
            color: rgba(0, 255, 136, 0.6);
            font-size: 13px;
            margin-top: 6px;
            letter-spacing: 1px;
        }}
        .stats {{
            background: linear-gradient(135deg, rgba(13, 17, 23, 0.95) 0%, rgba(22, 27, 34, 0.9) 100%);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            border: 1px solid rgba(0, 212, 255, 0.15);
            box-shadow: 
                0 4px 20px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }}
        .stat-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 26px;
            font-weight: 600;
            color: #00d4ff;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.4);
        }}
        .stat-label {{
            font-size: 11px;
            color: rgba(201, 209, 217, 0.5);
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        #graph {{
            width: 100%;
            height: calc(100vh - 200px);
            min-height: 500px;
            border-radius: 12px;
            background: rgba(8, 10, 14, 0.95);
            border: 1px solid rgba(0, 212, 255, 0.2);
            box-shadow: 
                0 8px 40px rgba(0, 0, 0, 0.5),
                inset 0 0 40px rgba(0, 212, 255, 0.03);
        }}
        .controls {{
            margin: 18px 0;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .controls button {{
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 0, 0, 0.4);
            color: #c9d1d9;
            border: 1px solid rgba(0, 212, 255, 0.3);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.25s ease;
            font-size: 13px;
        }}
        .controls button:hover {{
            background: rgba(0, 212, 255, 0.15);
            border-color: rgba(0, 212, 255, 0.5);
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
        }}
        .node-count {{
            font-family: 'JetBrains Mono', monospace;
            color: rgba(0, 255, 136, 0.5);
            font-size: 12px;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            align-items: center;
            font-size: 12px;
            color: rgba(201, 209, 217, 0.7);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-color {{
            width: 20px;
            height: 3px;
            border-radius: 2px;
        }}
        .legend-semantic {{ background: rgba(0, 212, 255, 0.7); }}
        .legend-temporal {{ background: rgba(255, 165, 0, 0.7); }}
        .legend-tag {{ background: rgba(138, 43, 226, 0.7); }}
        .top-nodes {{
            margin-top: 20px;
            background: linear-gradient(135deg, rgba(13, 17, 23, 0.95) 0%, rgba(22, 27, 34, 0.9) 100%);
            padding: 15px 20px;
            border-radius: 12px;
            border: 1px solid rgba(0, 212, 255, 0.15);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            max-height: 280px;
            overflow-y: auto;
        }}
        .top-nodes h3 {{
            font-family: 'JetBrains Mono', monospace;
            color: #00d4ff;
            margin: 0 0 15px 0;
            font-size: 14px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px 8px;
            text-align: left;
            border-bottom: 1px solid rgba(0, 212, 255, 0.08);
        }}
        th {{
            color: #00d4ff;
            font-weight: 500;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        td {{
            color: rgba(201, 209, 217, 0.8);
            font-size: 13px;
        }}
        tr:hover td {{
            background: rgba(0, 212, 255, 0.05);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="data:image/png;base64,{logo_base64}" class="logo" alt="Memora Logo">
            <div>
                <h1>MEMORA</h1>
                <div class="subtitle">// agent 记忆库</div>
            </div>
        </div>
    
    <div class="stats">
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{graph_data['stats']['total_nodes']:,}</div>
                <div class="stat-label">Total Nodes · 记忆总量</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{graph_data['stats']['graph_nodes']:,}</div>
                <div class="stat-label">Graph Nodes · 节点数</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{graph_data['stats']['graph_edges']:,}</div>
                <div class="stat-label">Connections · 连接数</div>
            </div>
        </div>
    </div>
    
    <div class="controls">
        <button onclick="fitGraph()">✧ 拟合视图</button>
        <button onclick="togglePhysics()">✦ 切换动态</button>
        <span class="node-count" id="nodeCount"></span>
        <div class="legend">
            <div class="legend-item"><div class="legend-color legend-semantic"></div>语义相似度</div>
            <div class="legend-item"><div class="legend-color legend-temporal"></div>时间相邻</div>
            <div class="legend-item"><div class="legend-color legend-tag"></div>共享标签</div>
        </div>
    </div>
    
    <div id="graph"></div>
    
    <div class="top-nodes">
        <h3>Top 10 PageRank Nodes</h3>
        <table>
            <tr>
                <th>Rank</th>
                <th>Node</th>
                <th>PR Score</th>
                <th>In</th>
                <th>Out</th>
            </tr>
"""

    # 添加前10节点
    for i, node in enumerate(graph_data['nodes'][:10], 1):
        html += f"""            <tr>
                <td>{i}</td>
                <td title="{node['id']}">{node['label']}</td>
                <td>{node['score']:.6f}</td>
                <td>{node['in_degree']}</td>
                <td>{node['out_degree']}</td>
            </tr>
"""

    html += """        </table>
    </div>
    
    <script>
        const nodes = new vis.DataSet(""" + json.dumps([{
            "id": n["id"],
            "label": n["label"],
            "value": float(n["score"]) * 2000,
            "title": "> PR: " + float(n["score"]).__format__(".6f") + "\\n> In: " + str(n["in_degree"]) + " / Out: " + str(n["out_degree"]),
            "font": { "color": "#c9d1d9", "size": 11 }
        } for n in graph_data["nodes"]]) + """);
        
        const edges = new vis.DataSet(""" + json.dumps([{
            "from": e["source"],
            "to": e["target"],
            "color": { 
                "color": {
                    "semantic": "rgba(0, 212, 255, 0.5)",   # 青色 - 语义相似度
                    "temporal": "rgba(255, 165, 0, 0.5)",   # 橙色 - 时间相邻
                    "tag": "rgba(138, 43, 226, 0.5)",       # 紫色 - 共享标签
                    "default": "rgba(128, 128, 128, 0.3)"   # 灰色 - 默认
                }.get(e.get("edge_type", "default"), "rgba(128, 128, 128, 0.3)")
            },
            "smooth": { "type": "continuous" }
        } for e in graph_data["edges"]]) + """);
        
        const container = document.getElementById("graph");
        const data = { nodes: nodes, edges: edges };
        const options = {
            nodes: {
                shape: "dot",
                borderWidth: 1.5,
                color: {
                    background: "rgba(0, 212, 255, 0.6)",
                    border: "rgba(0, 212, 255, 0.9)",
                    highlight: { background: "rgba(0, 255, 136, 0.9)", border: "#fff" },
                    hover: { background: "rgba(0, 212, 255, 0.9)", border: "#fff" }
                },
                shadow: {
                    enabled: true,
                    color: "rgba(0, 212, 255, 0.5)",
                    size: 12,
                    x: 0,
                    y: 0
                },
                scaling: { min: 4, max: 22 }
            },
            edges: {
                arrows: { to: { enabled: true, scaleFactor: 0.35, color: "rgba(0, 212, 255, 0.4)" } }
            },
            physics: {
                enabled: true,
                barnesHut: {
                    gravitationalConstant: -2500,
                    centralGravity: 0.25,
                    springLength: 150,
                    damping: 0.12
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 120,
                zoomView: true
            }
        };
        
        const network = new vis.Network(container, data, options);
        document.getElementById("nodeCount").textContent = nodes.length + " nodes / " + edges.length + " edges";
        
        function fitGraph() {
            network.fit({ animation: { duration: 600, easingFunction: "easeInOutQuad" }});
        }
        
        let physicsEnabled = true;
        function togglePhysics() {
            physicsEnabled = !physicsEnabled;
            network.setOptions({ physics: { enabled: physicsEnabled }});
        }
    </script>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML visualization saved to: {output_path}")


def generate_mermaid(graph_data: dict) -> str:
    """生成 Mermaid 图代码（文本格式）"""
    lines = ["```mermaid", "graph TD"]
    
    # 只显示top 50节点
    for node in graph_data['nodes'][:50]:
        safe_id = node['id'].replace('-', '_').replace('/', '_')[:20]
        lines.append(f"    {safe_id}[\"{node['label']}\\nPR:{node['score']:.4f}\"]")
    
    # 添加边
    for edge in graph_data['edges'][:100]:
        src = edge['source'].replace('-', '_').replace('/', '_')[:20]
        tgt = edge['target'].replace('-', '_').replace('/', '_')[:20]
        lines.append(f"    {src} --> {tgt}")
    
    lines.append("```")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate PageRank graph structure')
    parser.add_argument('--data', '-d', default='data', help='Data directory')
    parser.add_argument('--output', '-o', default='pagerank_graph.html', help='Output HTML file')
    parser.add_argument('--max-nodes', '-n', type=int, default=500, help='Max nodes to visualize')
    parser.add_argument('--text-only', '-t', action='store_true', help='Output text format only')
    args = parser.parse_args()
    
    print("📊 Loading memory nodes...")
    data_dir = Path(__file__).parent.parent / args.data
    nodes = load_all_nodes(str(data_dir))
    print(f"   Loaded {len(nodes)} nodes")
    
    print("📈 Building graph...")
    stats = build_graph_stats(nodes)
    print(f"   Nodes with links: {stats['nodes_with_links']}")
    print(f"   Total edges: {stats['total_edges']}")
    
    print("🔗 Generating graph data...")
    graph_data = generate_graph_data(nodes, max_nodes=args.max_nodes)
    
    # 输出统计
    print("\n" + "="*50)
    print("📊 PAGE RANK STATISTICS")
    print("="*50)
    print(f"Total Nodes: {stats['total_nodes']:,}")
    print(f"Nodes with Links: {stats['nodes_with_links']:,} ({100*stats['nodes_with_links']/stats['total_nodes']:.1f}%)")
    print(f"Total Edges: {stats['total_edges']:,}")
    
    # 显示边类型统计
    if 'edge_type_count' in graph_data:
        print(f"\n🔗 EDGE TYPE DISTRIBUTION:")
        edge_types = graph_data['edge_type_count']
        if edge_types.get('semantic', 0) > 0:
            print(f"  🔵 语义相似度: {edge_types.get('semantic', 0)} 条")
        if edge_types.get('temporal', 0) > 0:
            print(f"  🟠 时间相邻: {edge_types.get('temporal', 0)} 条")
        if edge_types.get('tag', 0) > 0:
            print(f"  🟣 共享标签: {edge_types.get('tag', 0)} 条")
    print(f"\nSource Distribution:")
    for source, count in stats['source_distribution'].items():
        print(f"  {source}: {count:,} nodes")
    
    # Top 10 PageRank
    print(f"\n🏆 TOP 10 PAGERANK NODES:")
    for i, node in enumerate(graph_data['nodes'][:10], 1):
        print(f"  {i:2d}. {node['label'][:40]:<40} PR={node['score']:.6f} in={node['in_degree']} out={node['out_degree']}")
    
    # 生成输出
    if args.text_only:
        print("\n📝 MERMAID GRAPH:")
        print(generate_mermaid(graph_data))
    else:
        output_path = Path(__file__).parent.parent / args.output
        generate_html_visualization(graph_data, str(output_path))
        print(f"\n🌐 Open in browser: file://{output_path}")
    
    # 同时输出 mermaid
    print("\n📝 MERMAID CODE:")
    print(generate_mermaid(graph_data))


if __name__ == "__main__":
    main()
