# embed.py
import os
import chunk
import chromadb
from zhipuai import ZhipuAI

# 从环境变量获取API密钥
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 ZHIPUAI_API_KEY")


client = ZhipuAI(api_key=api_key)
EMBEDDING_MODEL = "embedding-3"  # 使用智谱AI的嵌入模型
LLM_MODEL = "glm-4.6v-flash"  # GLM-4.6V-Flash模型

chromadb_client = chromadb.PersistentClient("./chroma.db")
chromadb_collection = chromadb_client.get_or_create_collection("linghuchong")

def embed(text: str, store: bool) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def create_db() -> None:
    for idx, c in enumerate(chunk.get_chunks()):
        print(f"Process: {c}")
        embedding = embed(c, store=True)
        chromadb_collection.upsert(
            ids=str(idx),
            documents=c,
            embeddings=embedding
        )

def query_db(question: str) -> list[str]:
    question_embedding = embed(question, store=False)
    result = chromadb_collection.query(
        query_embeddings=[question_embedding],  # 注意：嵌入向量需要作为列表传递
        n_results=5
    )
    assert result["documents"]
    return result["documents"][0]


if __name__ == '__main__':
    # question = "令狐冲领悟了什么魔法？"
    question = "大连是哪个省的城市？"
    # create_db()
    chunks = query_db(question)
    prompt = "Please answer user's question according to context\n"
    prompt += f"Question: {question}\n"
    prompt += "Context:\n"
    for c in chunks:
        prompt += f"{c}\n"
        prompt += "-------------\n"
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    print(response.choices[0].message.content)