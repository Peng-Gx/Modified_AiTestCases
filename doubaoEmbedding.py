import requests
import numpy as np
from langchain_core.embeddings import Embeddings

class DoubaoEmbedding(Embeddings):
    def __init__(self, api_url: str, api_key: str, model_name: str = "doubao-embedding-large"):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    def embed_documents(self, documents: list) -> np.ndarray:
        # 设置请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 请求数据体
        data = {
            "model": self.model_name,  # 模型名称
            "input": documents,        # 要嵌入的文本列表
            "encode_type": "text2vec"  # 假设是文本到向量的转换类型
        }

        # 发送 POST 请求到 API
        response = requests.post(self.api_url, json=data, headers=headers)

        if response.status_code != 200:
            raise Exception(f"API 请求失败: {response.text}")

        # 解析 JSON 响应
        response_json = response.json()
        # print(response_json)

        if "data" not in response_json:
            raise ValueError(f"Unexpected response format: {data}")

        embeddings = []
        for result in response_json["data"]:
            if "embedding" not in result:
                raise ValueError(f"Missing 'embedding' in result: {result}")
            embeddings.append(result["embedding"])

        return np.array(embeddings)


    def embed_query(self, query: str) -> np.ndarray:
        # 如果只需要处理单个查询，调用 `embed_documents` 即可
        return self.embed_documents([query])[0]