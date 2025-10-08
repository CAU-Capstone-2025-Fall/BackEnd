import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # ✅ .env 파일 불러오기

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
INPUT_PATH = "data/data.json"
OUTPUT_PATH = "data/multi_embedded.json"

def embed_grouped_queries():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []

    for item in data:
        queries = item.get("queries", [])
        answer = item["answer"]

        if not queries:
            continue

        print(f"▶ {len(queries)}개 쿼리 임베딩 중...")

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=queries
        )

        embedded_queries = [r.embedding for r in response.data]

        result.append({
            "queries": queries,
            "embedded_queries": embedded_queries,
            "answer": answer
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료: {len(result)}개 그룹 → {OUTPUT_PATH}")

if __name__ == "__main__":
    embed_grouped_queries()
    embed_grouped_queries()
