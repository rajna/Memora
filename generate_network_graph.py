#!/usr/bin/env python3
"""
Memora 记忆网络图生成器
读取markdown文件，生成可拖拽的可视化网络图
"""

import os
import re
import json
import frontmatter
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def extract_node_from_md(filepath: str) -> Dict[str, Any]:
    """从markdown文件提取节点数据"""
    post = frontmatter.load(filepath)
    
    # 提取内容（去掉frontmatter后的正文）
    content = post.content.strip()
    # 截断过长内容
    if len(content) > 300:
        content = content[:300] + "..."
    
    # 清理内容中的特殊字符，避免破坏HTML/JS
    content = content.replace("'", "\\'").replace('`', '\\`')
    content = content.replace('\n', ' ').replace('\r', '')
    content = content.replace('<', '&lt;').replace('>', '&gt;')
    
    # 提取标题并清理
    title = post.get('title', '无标题')
    title = title.replace("'", "\\'").replace('`', '\\`')
    
    # 提取元数据
    node = {
        'id': post.get('id', ''),
        'title': title,
        'content': content,
        'pagerank': post.get('pagerank', 0.01),
        'tags': post.get('tags', []),
        'created': post.get('created', ''),
        'links': post.get('links', []),
        'backlinks': post.get('backlinks', []),
        'url': post.get('url', '')
    }
    
    # 格式化创建时间
    if node['created']:
        try:
            dt = datetime.fromisoformat(node['created'].replace('Z', '+00:00'))
            node['created'] = dt.strftime('%Y-%m-%d %H:%M')
        except:
            pass
    
    return node


def build_semantic_links(nodes: List[Dict]) -> List[Dict]:
    """基于共享标签构建语义相似度链接"""
    edges = []
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            # 计算共享标签
            shared_tags = set(node1['tags']) & set(node2['tags'])
            
            if len(shared_tags) >= 3:  # 至少3个共享标签（降低密度）
                weight = min(0.5 + len(shared_tags) * 0.1, 0.85)
                edges.append({
                    'from': node1['id'],
                    'to': node2['id'],
                    'label': f'PR:{weight:.2f}',
                    'color': '#29b6f6',
                    'width': 1 + weight * 1.5,
                    'type': 'semantic',
                    'dashes': False,
                    'arrows': False,
                    'visible': False  # 默认不显示，避免干扰
                })
    
    return edges


def build_temporal_links(nodes: List[Dict]) -> List[Dict]:
    """基于创建时间构建时间相邻链接"""
    # 按时间排序
    sorted_nodes = sorted(nodes, key=lambda x: x['created'] or '')
    edges = []
    
    for i in range(len(sorted_nodes) - 1):
        edges.append({
            'from': sorted_nodes[i]['id'],
            'to': sorted_nodes[i+1]['id'],
            'label': '',
            'color': '#ff5252',
            'width': 2,
            'type': 'temporal',
            'dashes': False,
            'arrows': True,
            'visible': True
        })
    
    return edges


def build_tag_links(nodes: List[Dict]) -> List[Dict]:
    """构建共享标签链接（标签重叠度高的）"""
    edges = []
    processed_pairs = set()
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            pair_key = tuple(sorted([node1['id'], node2['id']]))
            if pair_key in processed_pairs:
                continue
                
            shared_tags = set(node1['tags']) & set(node2['tags'])
            
            # 如果共享1-2个标签，添加标签链接
            if 1 <= len(shared_tags) < 2:
                weight = 0.3 + len(shared_tags) * 0.2
                edges.append({
                    'from': node1['id'],
                    'to': node2['id'],
                    'label': f'PR:{weight:.2f}',
                    'color': '#ab47bc',
                    'width': len(shared_tags),
                    'type': 'tag',
                    'dashes': False,
                    'arrows': False
                })
                processed_pairs.add(pair_key)
    
    return edges


def generate_html(nodes: List[Dict], edges: List[Dict], output_path: str):
    """生成HTML文件"""
    
    # 将节点和边转为JSON
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    
    # 构建HTML - 使用模板字符串避免换行问题
    html_parts = []
    html_parts.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memora 记忆网络 · ''' + str(len(nodes)) + '''节点可视化</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #415a77 100%);
            min-height: 100vh;
            overflow: hidden;
        }
        #mynetwork { width: 100vw; height: 100vh; position: relative; }
        
        /* 🌟 顶部标题栏 */
        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 3000;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(180deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0) 100%);
            pointer-events: none;
        }
        .header-content {
            display: flex;
            align-items: center;
            gap: 20px;
            pointer-events: auto;
        }
        .logo-container {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4), 
                        0 0 60px rgba(102, 126, 234, 0.2),
                        inset 0 1px 0 rgba(255,255,255,0.2);
            position: relative;
            overflow: hidden;
        }
        .logo-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 60%);
            animation: shimmer 3s ease-in-out infinite;
        }
        @keyframes shimmer {
            0%, 100% { transform: rotate(0deg); }
            50% { transform: rotate(180deg); }
        }
        .logo-img {
            width: 44px;
            height: 44px;
            object-fit: contain;
            z-index: 1;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.4));
            border-radius: 8px;
        }
        .title-section {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .main-title {
            font-size: 28px;
            font-weight: 700;
            color: #fff;
            letter-spacing: 2px;
            text-shadow: 0 2px 20px rgba(102, 126, 234, 0.5);
            background: linear-gradient(90deg, #fff 0%, #a8c6fa 50%, #fff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            font-size: 13px;
            color: rgba(255,255,255,0.6);
            letter-spacing: 3px;
            font-weight: 400;
        }
        .tagline {
            font-size: 12px;
            color: rgba(255,255,255,0.4);
            font-style: italic;
            margin-top: 2px;
        }
        
        .node-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 12px 16px;
            width: 260px;
            max-height: 180px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            cursor: grab;
            user-select: none;
            position: absolute;
            transition: box-shadow 0.2s, transform 0.1s;
            z-index: 10;
        }
        .node-card:active { cursor: grabbing; box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5); transform: scale(1.02); }
        .node-card.dragging { opacity: 0.9; z-index: 1000 !important; }
        .node-card:hover { background: rgba(255, 255, 255, 0.12); border-color: rgba(255, 255, 255, 0.25); }
        .node-card .title {
            font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 8px;
            line-height: 1.4; border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 6px; pointer-events: none;
        }
        .node-card .content {
            font-size: 11px; color: rgba(255, 255, 255, 0.75); line-height: 1.5;
            overflow: hidden; display: -webkit-box; -webkit-line-clamp: 5;
            -webkit-box-orient: vertical; pointer-events: none;
        }
        .node-card .meta {
            margin-top: 8px; font-size: 10px; color: rgba(255, 255, 255, 0.5);
            display: flex; justify-content: space-between; align-items: center; pointer-events: none;
        }
        .node-card .pr-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 2px 8px; border-radius: 10px; font-size: 9px; font-weight: 600;
        }
        .node-card .tags { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 6px; pointer-events: none; }
        .node-card .tag { background: rgba(255, 255, 255, 0.1); color: rgba(255, 255, 255, 0.7); padding: 2px 6px; border-radius: 4px; font-size: 9px; }
        .controls {
            position: fixed; top: 20px; right: 20px; background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px); border-radius: 12px; padding: 16px;
            color: white; font-size: 12px; z-index: 2000;
            border: 1px solid rgba(255, 255, 255, 0.1); min-width: 180px;
        }
        .controls h3 { margin-bottom: 12px; font-size: 14px; color: #fff; }
        .legend-item { display: flex; align-items: center; margin-bottom: 8px; cursor: pointer; opacity: 0.9; }
        .legend-item:hover { opacity: 1; }
        .legend-item.disabled { opacity: 0.3; }
        .legend-color { width: 24px; height: 3px; margin-right: 8px; border-radius: 2px; }
        .legend-semantic { background: #4fc3f7; }
        .legend-temporal { background: #ff7043; }
        .legend-tag { background: #ab47bc; }
        .stats { margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.1); }
        .stat-item { display: flex; justify-content: space-between; margin-bottom: 4px; }
        .detail-panel {
            position: fixed; left: 20px; top: 20px; width: 400px; max-height: calc(100vh - 40px);
            background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(20px); border-radius: 16px;
            padding: 20px; color: white; overflow-y: auto; z-index: 2000;
            border: 1px solid rgba(255, 255, 255, 0.15);
            transform: translateX(-120%); transition: transform 0.3s ease;
        }
        .detail-panel.active { transform: translateX(0); }
        .detail-panel .close-btn {
            position: absolute; top: 12px; right: 12px; width: 28px; height: 28px;
            background: rgba(255, 255, 255, 0.1); border: none; border-radius: 50%;
            color: white; cursor: pointer; font-size: 16px;
            display: flex; align-items: center; justify-content: center;
        }
        .detail-panel .close-btn:hover { background: rgba(255, 255, 255, 0.2); }
        .detail-panel h2 { font-size: 16px; margin-bottom: 12px; padding-right: 30px; line-height: 1.4; }
        .detail-panel .meta-info { font-size: 11px; color: rgba(255, 255, 255, 0.6); margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
        .detail-panel .content-full { font-size: 13px; line-height: 1.8; color: rgba(255, 255, 255, 0.9); }
        #connections-layer { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1; }
        .hint { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(0, 0, 0, 0.6); backdrop-filter: blur(8px); color: rgba(255, 255, 255, 0.8); padding: 10px 20px; border-radius: 20px; font-size: 12px; z-index: 2000; pointer-events: none; }
    </style>
</head>
<body>
    <!-- 🌟 Memora 品牌标题 -->
    <div class="header">
        <div class="header-content">
            <div class="logo-container">
                <img src="logo.png" alt="Memora Logo" class="logo-img">
            </div>
            <div class="title-section">
                <div class="main-title">MEMORA</div>
                <div class="subtitle">AI AGENT 记忆系统</div>
                <div class="tagline">✨ Memory + Aurora — 记忆极光，让知识自由流动</div>
            </div>
        </div>
    </div>
    
    <div id="mynetwork"><svg id="connections-layer"></svg></div>
    <div class="controls">
        <h3>🔗 链接类型</h3>
        <div class="legend-item disabled" data-type="semantic"><div class="legend-color legend-semantic"></div><span>语义相似度</span></div>
        <div class="legend-item" data-type="temporal"><div class="legend-color legend-temporal"></div><span>时间相邻</span></div>
        <div class="legend-item disabled" data-type="tag"><div class="legend-color legend-tag"></div><span>共享标签</span></div>
        <div class="stats">
            <div class="stat-item"><span>节点数:</span><span>''' + str(len(nodes)) + '''</span></div>
            <div class="stat-item"><span>链接数:</span><span>''' + str(len(edges)) + '''</span></div>
            <div class="stat-item"><span>平均PR:</span><span>''' + f"{sum(n.get('pagerank', 0.01) for n in nodes)/len(nodes):.3f}" + '''</span></div>
        </div>
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
            <button onclick="autoLayout()" style="background: rgba(102, 126, 234, 0.8); color: white; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 11px; width: 100%;">🔄 自动布局</button>
        </div>
    </div>
    <div class="detail-panel" id="detailPanel">
        <button class="close-btn" onclick="closeDetail()">×</button>
        <h2 id="detailTitle">-</h2>
        <div class="meta-info" id="detailMeta">-</div>
        <div class="content-full" id="detailContent">-</div>
    </div>
    <div class="hint">💡 拖拽卡片调整位置，点击查看详情</div>
    <script>
''')
    
    # JavaScript 代码部分
    js_code = f'''
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        const cardElements = {{}};
        let draggedCard = null;
        let dragOffset = {{ x: 0, y: 0 }};

        function init() {{
            const container = document.getElementById('mynetwork');
            nodesData.forEach((node, index) => {{
                const card = createCard(node);
                const angle = (index / nodesData.length) * 2 * Math.PI;
                const radius = Math.min(window.innerWidth, window.innerHeight) * 0.35;
                const centerX = window.innerWidth / 2;
                const centerY = window.innerHeight / 2;
                card.style.left = (centerX + Math.cos(angle) * radius - 130) + 'px';
                card.style.top = (centerY + Math.sin(angle) * radius - 90) + 'px';
                container.appendChild(card);
                cardElements[node.id] = card;
                bindDragEvents(card, node);
            }});
            drawConnections();
            window.addEventListener('resize', drawConnections);
        }}

        function createCard(node) {{
            const div = document.createElement('div');
            div.className = 'node-card';
            div.dataset.nodeId = node.id;
            const tagsHtml = (node.tags || []).slice(0, 3).map(t => `<span class="tag">${{t}}</span>`).join('');
            div.innerHTML = `<div class="title">${{node.title}}</div><div class="content">${{node.content}}</div><div class="meta"><span>${{node.created || '未知时间'}}</span><span class="pr-badge">PR: ${{parseFloat(node.pagerank || 0.01).toFixed(2)}}</span></div><div class="tags">${{tagsHtml}}</div>`;
            div.addEventListener('click', function(e) {{
                if (!div.classList.contains('was-dragged')) showDetail(node);
                div.classList.remove('was-dragged');
            }});
            return div;
        }}

        function bindDragEvents(card, node) {{
            card.addEventListener('mousedown', function(e) {{
                if (e.button !== 0) return;
                draggedCard = card;
                card.classList.add('dragging');
                const rect = card.getBoundingClientRect();
                dragOffset.x = e.clientX - rect.left;
                dragOffset.y = e.clientY - rect.top;
                e.preventDefault();
            }});
        }}

        document.addEventListener('mousemove', function(e) {{
            if (!draggedCard) return;
            draggedCard.style.left = (e.clientX - dragOffset.x) + 'px';
            draggedCard.style.top = (e.clientY - dragOffset.y) + 'px';
            draggedCard.classList.add('was-dragged');
            drawConnections();
        }});

        document.addEventListener('mouseup', function() {{
            if (draggedCard) {{
                draggedCard.classList.remove('dragging');
                draggedCard = null;
            }}
        }});

        function drawConnections() {{
            const svg = document.getElementById('connections-layer');
            svg.innerHTML = '';
            const visibleTypes = getVisibleTypes();
            edgesData.forEach(edge => {{
                if (!visibleTypes.includes(edge.type)) return;
                const fromCard = cardElements[edge.from];
                const toCard = cardElements[edge.to];
                if (!fromCard || !toCard) return;
                const fromRect = fromCard.getBoundingClientRect();
                const toRect = toCard.getBoundingClientRect();
                const x1 = fromRect.left + fromRect.width / 2;
                const y1 = fromRect.top + fromRect.height / 2;
                const x2 = toRect.left + toRect.width / 2;
                const y2 = toRect.top + toRect.height / 2;
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', x1); line.setAttribute('y1', y1);
                line.setAttribute('x2', x2); line.setAttribute('y2', y2);
                line.setAttribute('stroke', edge.color);
                line.setAttribute('stroke-width', edge.width);
                line.setAttribute('stroke-opacity', edge.type === 'temporal' ? '0.4' : '0.6');
                if (edge.dashes) line.setAttribute('stroke-dasharray', '5,5');
                svg.appendChild(line);
                if (edge.label) {{
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', (x1 + x2) / 2);
                    text.setAttribute('y', (y1 + y2) / 2 - 5);
                    text.setAttribute('text-anchor', 'middle');
                    text.setAttribute('fill', '#ffd700');
                    text.setAttribute('font-size', '10');
                    text.setAttribute('font-weight', '600');
                    text.setAttribute('style', 'text-shadow: 0 1px 2px rgba(0,0,0,0.8);');
                    text.textContent = edge.label;
                    svg.appendChild(text);
                }}
                if (edge.arrows) {{
                    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    const angle = Math.atan2(y2 - y1, x2 - x1);
                    const arrowLength = 10, arrowAngle = 0.5;
                    const targetX = x2 - Math.cos(angle) * (toRect.width / 2 + 5);
                    const targetY = y2 - Math.sin(angle) * (toRect.height / 2 + 5);
                    const ax1 = targetX - arrowLength * Math.cos(angle - arrowAngle);
                    const ay1 = targetY - arrowLength * Math.sin(angle - arrowAngle);
                    const ax2 = targetX - arrowLength * Math.cos(angle + arrowAngle);
                    const ay2 = targetY - arrowLength * Math.sin(angle + arrowAngle);
                    arrow.setAttribute('points', `${{targetX}},${{targetY}} ${{ax1}},${{ay1}} ${{ax2}},${{ay2}}`);
                    arrow.setAttribute('fill', edge.color);
                    arrow.setAttribute('opacity', '0.6');
                    svg.appendChild(arrow);
                }}
            }});
        }}

        function getVisibleTypes() {{
            const types = [];
            document.querySelectorAll('.legend-item').forEach(item => {{
                if (!item.classList.contains('disabled')) types.push(item.dataset.type);
            }});
            return types;
        }}

        document.querySelectorAll('.legend-item').forEach(item => {{
            item.addEventListener('click', function() {{
                this.classList.toggle('disabled');
                drawConnections();
            }});
        }});

        function showDetail(node) {{
            const panel = document.getElementById('detailPanel');
            document.getElementById('detailTitle').textContent = node.title;
            document.getElementById('detailMeta').innerHTML = `创建时间: ${{node.created || '未知'}} | PR值: ${{parseFloat(node.pagerank || 0.01).toFixed(2)}}<br>标签: ${{(node.tags || []).join(', ') || '无'}}<br>外链: ${{(node.links || []).length}} | 反链: ${{(node.backlinks || []).length}}`;
            let content = node.content || '';
            if (content.length > 500 && !content.endsWith('...')) content += '...';
            const fullContent = content + '\\n\\n---\\n\\n**标签**: ' + (node.tags || []).map(t => '`' + t + '`').join(' ') + '\\n\\n**链接**: ' + ((node.links || []).length > 0 ? (node.links || []).slice(0, 5).join(', ') + ((node.links || []).length > 5 ? '...' : '') : '无');
            document.getElementById('detailContent').innerHTML = marked.parse(fullContent);
            panel.classList.add('active');
        }}

        function closeDetail() {{
            document.getElementById('detailPanel').classList.remove('active');
        }}

        function autoLayout() {{
            const centerX = window.innerWidth / 2;
            const centerY = window.innerHeight / 2;
            const radius = Math.min(window.innerWidth, window.innerHeight) * 0.35;
            nodesData.forEach((node, index) => {{
                const card = cardElements[node.id];
                if (card) {{
                    const angle = (index / nodesData.length) * 2 * Math.PI;
                    card.style.transition = 'left 0.5s ease, top 0.5s ease';
                    card.style.left = (centerX + Math.cos(angle) * radius - 130) + 'px';
                    card.style.top = (centerY + Math.sin(angle) * radius - 90) + 'px';
                    setTimeout(() => {{ card.style.transition = ''; drawConnections(); }}, 500);
                }}
            }});
            setTimeout(drawConnections, 100);
        }}

        init();
'''
    
    html_parts.append(js_code)
    html_parts.append('    </script>\n</body>\n</html>')
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    print(f"✅ 已生成: {output_path}")
    print(f"   节点数: {len(nodes)}")
    print(f"   链接数: {len(edges)}")
    print(f"   语义链接: {sum(1 for e in edges if e['type'] == 'semantic')}")
    print(f"   时间链接: {sum(1 for e in edges if e['type'] == 'temporal')}")
    print(f"   标签链接: {sum(1 for e in edges if e['type'] == 'tag')}")


def main():
    """主函数"""
    file_paths = [
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604050140-9ce90e3d.md',
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604051449-64ef5d5b.md',
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604051148-7d1fd8e7.md',
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604051230-64765a27.md',
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604050021-5faa7fd8.md',
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604050012-876bd5ab.md',
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604050029-75233df4.md',
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604051911-502494e9.md',
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604050035-35dc0024.md',
        '/Users/rama/.nanobot/workspace/Memora/data/2026/04/05/202604050347-0657f90b.md',
    ]
    
    print("📖 读取节点数据...")
    nodes = []
    for fp in file_paths:
        if os.path.exists(fp):
            node = extract_node_from_md(fp)
            nodes.append(node)
            print(f"   ✓ {node['id'][:20]}... - {node['title'][:30]}")
        else:
            print(f"   ✗ 文件不存在: {fp}")
    
    if not nodes:
        print("❌ 没有读取到任何节点")
        return
    
    print("\n🔗 构建链接关系...")
    edges = []
    edges.extend(build_semantic_links(nodes))
    edges.extend(build_temporal_links(nodes))
    edges.extend(build_tag_links(nodes))
    
    output_path = '/Users/rama/.nanobot/workspace/Memora/network_graph_10nodes.html'
    print("\n🎨 生成可视化网页...")
    generate_html(nodes, edges, output_path)
    
    print(f"\n🚀 打开网页:")
    print(f"   open {output_path}")


if __name__ == '__main__':
    main()
