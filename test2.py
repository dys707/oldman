import os
import chromadb
import pandas as pd


def inspect_chroma_vector_db(db_path="vector_db", collection_name=None):
    """
    增强版：读取 Chroma 向量数据库并以表格形式展示
    注意：db_path 传 chroma_store 所在的文件夹（比如 vector_db），不是直接传 .sqlite3 文件！
    """
    # 检查路径是否存在（先转绝对路径，避免相对路径坑）
    abs_path = os.path.abspath(db_path)
    print(f"🔍 检查数据库路径: {abs_path}")

    if not os.path.exists(abs_path):
        print(f"❌ 路径不存在: {abs_path}")
        # 尝试自动补全 chroma_store（常见路径）
        candidate_path = os.path.join(db_path, "chroma_store")
        if os.path.exists(candidate_path):
            print(f"🔄 尝试自动修正路径为: {candidate_path}")
            abs_path = candidate_path
        else:
            return

    # 列出路径下的文件（确认 chroma.sqlite3 存在）
    print(f"📁 路径下的内容:")
    for item in os.listdir(abs_path):
        item_path = os.path.join(abs_path, item)
        print(f"  - {item} {'(文件夹)' if os.path.isdir(item_path) else ''}")

    # 检查 chroma.sqlite3 是否存在
    sqlite_path = os.path.join(abs_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        print(f"❌ 路径下未找到 chroma.sqlite3: {sqlite_path}")
        return

    # 1. 初始化 Chroma 客户端（关键：传文件夹路径，不是 .sqlite3 文件！）
    try:
        client = chromadb.PersistentClient(path=abs_path)
        print("✅ 客户端初始化成功")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return

    # 2. 列出所有集合
    try:
        collections = client.list_collections()
        print(f"📚 找到 {len(collections)} 个集合：")
        for coll in collections:
            print(f"  - {coll.name}")
            # 查看每个集合的数据量
            count = coll.count()
            print(f"    数据条数: {count}")
    except Exception as e:
        print(f"❌ 获取集合失败: {e}")
        return

    print("-" * 80)

    if not collections:
        print("❌ 没有任何集合，数据库是空的")
        return

    # 3. 如果没指定集合名，默认用第一个集合
    if collection_name is None:
        collection_name = collections[0].name
        print(f"ℹ️  默认查看第一个集合：{collection_name}")

    # 4. 获取集合
    try:
        collection = client.get_collection(name=collection_name)
        print(f"✅ 成功获取集合: {collection_name}")

        # 查看集合中的数据量
        count = collection.count()
        print(f"📊 集合中的数据条数: {count}")

        if count == 0:
            print("⚠️  集合存在但数据条数为0")

    except Exception as e:
        print(f"❌ 获取集合 {collection_name} 失败：{e}")
        return

    # 5. 读取所有数据
    try:
        data = collection.get(
            include=["metadatas", "documents", "embeddings"]
        )

        print(f"📦 获取到的数据: ids数量={len(data['ids'])}")

    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return

    if not data["ids"]:
        print("⚠️ 集合存在，但 get() 返回的数据为空")

        # 尝试用 limit 方式获取
        print("🔄 尝试用 limit 方式获取数据...")
        try:
            data = collection.get(limit=10, include=["metadatas", "documents"])
            if data["ids"]:
                print(f"✅ limit 方式获取到 {len(data['ids'])} 条数据")
            else:
                print("❌ limit 方式也没有数据")
        except Exception as e:
            print(f"❌ limit 方式失败: {e}")

        return

    # 6. 转成 DataFrame 并格式化展示
    df_data = {
        "id": data["ids"],
        "document": [doc[:100] + "..." if doc and len(doc) > 100 else doc for doc in data["documents"]],
    }

    # 处理 metadatas
    if data["metadatas"] and len(data["metadatas"]) > 0:
        # 获取所有可能的 metadata 键
        all_keys = set()
        for meta in data["metadatas"]:
            if meta:
                all_keys.update(meta.keys())

        # 为每个键创建列
        for key in all_keys:
            df_data[key] = [meta.get(key, None) if meta else None for meta in data["metadatas"]]

    df = pd.DataFrame(df_data)

    print(f"\n📊 集合「{collection_name}」的内容（共 {len(df)} 条）：")
    if len(df) > 0:
        print(df.to_string(index=False))
    else:
        print("DataFrame 为空")

    # 7. 可选：打印前几条 embedding（向量）预览
    if data.get("embeddings") and len(data["embeddings"]) > 0:
        print(f"\n🧠 前 2 条 embedding 预览（维度：{len(data['embeddings'][0])}）：")
        for i, emb in enumerate(data["embeddings"][:2]):
            emb_str = str(emb[:5]) + "..." if len(emb) > 5 else str(emb)
            print(f"  [{i}] {emb_str}")
    else:
        print("\n⚠️ 没有 embedding 数据")


if __name__ == "__main__":
    # 修复点1：possible_paths 要写成列表，不是字符串！
    # 这里填你实际的 vector_db 路径，比如：
    # Windows: ["C:\\你的项目路径\\vector_db"]
    # Mac/Linux: ["/Users/你的用户名/项目路径/vector_db"]
    possible_paths = ["vector_db"]  # 脚本和 vector_db 同目录就用这个

    for path in possible_paths:
        print(f"\n{'=' * 60}")
        print(f"尝试路径: {path}")
        print('=' * 60)
        inspect_chroma_vector_db(db_path=path)