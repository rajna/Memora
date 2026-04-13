import json
import sys
sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem

input_file = "/Users/rama/.nanobot/workspace/Memora/tools/cli_direct_2026-03-10.jsonl"

# 加载所有会话
conversations = []
current_session = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            role = msg.get('role', '')
            if role == 'user':
                if current_session:
                    conversations.append(current_session)
                current_session = [msg]
            elif role in ('assistant', 'tool') and current_session:
                current_session.append(msg)
        except:
            pass
if current_session:
    conversations.append(current_session)

print(f"总共 {len(conversations)} 组对话")

# 初始化 MemorySystem
ms = MemorySystem()

# 导入剩余的（跳过前5组）
imported = 0
for i, session in enumerate(conversations[5:], 6):
    try:
        node = ms.add_memory_from_messages(
            messages=session,
            source="cli-import",
            base_tags=["cli-import", "2026-03-10"]
        )
        if node:
            imported += 1
            if i % 3 == 0:
                print(f"  进度: {i}/{len(conversations)}")
    except Exception as e:
        print(f"  失败 {i}: {e}")

print(f"\n导入完成: {imported} 组")
print("开始构建图...")
ms.build_graph(auto_link=True)

stats = ms.stats()
print(f"\n✅ 完成!")
print(f"   总节点数: {stats['total_nodes']}")
print(f"   总标签数: {stats['total_tags']}")
