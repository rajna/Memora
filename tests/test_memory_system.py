"""
Memory System Tests
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tempfile
import shutil
from datetime import datetime

from memory_system import MemorySystem
from models import MemoryNode


def test_add_and_search():
    """测试添加和搜索记忆"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        ms = MemorySystem(
            memory_dir=os.path.join(temp_dir, "memories"),
            embedding_dir=os.path.join(temp_dir, "embeddings")
        )
        
        # 添加记忆
        node1 = ms.add_memory(
            content="VRM动捕项目使用MediaPipe进行姿态估计",
            title="VRM Motion Capture",
            tags=["project", "vrm"]
        )
        
        node2 = ms.add_memory(
            content="灵痕小说写作技巧分析",
            title="灵痕 Novel",
            tags=["project", "novel"]
        )
        
        assert node1.id is not None
        assert node1.url.startswith("/memory/")
        
        # 搜索
        results = ms.search("VRM", top_k=5)
        assert len(results) > 0
        assert results[0].node.title == "VRM Motion Capture"
        
        print("✓ test_add_and_search passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_pagerank():
    """测试PageRank计算"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        ms = MemorySystem(
            memory_dir=os.path.join(temp_dir, "memories"),
            embedding_dir=os.path.join(temp_dir, "embeddings")
        )
        
        # 添加多个记忆
        nodes = []
        for i in range(5):
            node = ms.add_memory(
                content=f"Test memory content {i}",
                title=f"Test {i}",
                tags=["test"]
            )
            nodes.append(node)
        
        # 手动建立链接
        ms.link_memories(nodes[0].url, nodes[1].url)
        ms.link_memories(nodes[0].url, nodes[2].url)
        ms.link_memories(nodes[1].url, nodes[3].url)
        
        # 构建图谱
        scores = ms.build_graph(auto_link=False)
        
        # nodes[0]有更多外链，应该PageRank更高
        assert scores[nodes[0].url] > scores[nodes[3].url]
        
        print("✓ test_pagerank passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_storage():
    """测试存储功能"""
    from storage import MemoryStorage
    from models import MemoryNode
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        storage = MemoryStorage(temp_dir)
        
        node = MemoryNode(
            id="test-id",
            url="/memory/2026/04/08/test",
            created=datetime.now(),
            modified=datetime.now(),
            content="Test content",
            title="Test",
            tags=["test"]
        )
        
        # 保存
        path = storage.save(node)
        assert os.path.exists(path)
        
        # 加载
        loaded = storage.load(path)
        assert loaded.id == node.id
        assert loaded.content == node.content
        
        # 通过URL加载
        loaded_by_url = storage.load_by_url(node.url)
        assert loaded_by_url is not None
        
        print("✓ test_storage passed")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_storage()
    test_add_and_search()
    test_pagerank()
    print("\nAll tests passed!")
