import json
import sys
sys.path.insert(0, '/Users/rama/.nanobot/workspace/Memora')
from src.memory_system import MemorySystem

input_file = "/Users/rama/.nanobot/workspace/Memora/tools/cli_direct_2026-03-10.jsonl"

# 加载所有会话
conversations = []
current_session = []
print("1. 正在加载会话数据...")
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

print(f"   总共 {len(conversations)} 组对话")
print(f"   需要导入: 第 6 组到第 {len(conversations)} 组 (共 {len(conversations)-5} 组)")

# 初始化 MemorySystem
print("\n2. 正在初始化 MemorySystem...")
ms = MemorySystem()

# 导入剩余的（跳过前5组）
print("\n3. 开始导入剩余会话...")
imported = 0
failed = []
for i, session in enumerate(conversations[5:], 6):
    try:
        node = ms.add_memory_from_messages(
            messages=session,
            source="cli-import",
            base_tags=["cli-import", "2026-03-10"]
        )
        if node:
            imported += 1
            print(f"   ✓ {i}/{len(conversations)}")
        else:
            failed.append(i)
            print(f"   ✗ {i}/{len(conversations)} (创建节点失败)")
    except Exception as e:
        failed.append(i)
        print(f"   ✗ {i}/{len(conversations)}: {str(e)[:50]}")

print(f"\n   导入完成: {imported} 组成功, {len(failed)} 组失败")
if failed:
    print(f"   失败的组: {failed}")

# 保存但不构建图（避免超时）
print("\n4. 正在保存索引...")
ms._save_index()

# 获取统计信息
stats = ms.stats()
print(f"\n✅ 完成!")
print(f"   本次导入: {imported} 组")
print(f"   总节点数: {stats['total_nodes']}")
print(f"   总标签数: {stats['total_tags']}")
print(f"\n⚠️ 注意: 图构建已跳过，可稍后手动运行 ms.build_graph(auto_link=True)")
