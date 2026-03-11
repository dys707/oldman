#纯调试代码
import chromadb
from src.app.utils import load_config

config = load_config()
client = chromadb.PersistentClient(path=config['vector_db_path'])

# 1. 先确认集合存在
collections = [c.name for c in client.list_collections()]
print("当前集合：", collections)

if "user_memory" in collections:
    # 2. 执行删除
    client.delete_collection("user_memory")
    print("✅ user_memory 集合已删除")
else:
    print("⚠️ user_memory 集合不存在")

# 3. 验证删除结果
new_collections = [c.name for c in client.list_collections()]
print("删除后集合：", new_collections)