import os
from config import (
    VOLCENGINE_API_SECRET
)
from openai import OpenAI
client = OpenAI(
    api_key = VOLCENGINE_API_SECRET,
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)
# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    model = "deepseek-r1-250528",  # your model endpoint ID
    messages = [
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
)
print(completion.choices[0].message.content)
# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
    model = "deepseek-r1-250528",  # your model endpoint ID
    messages = [
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
    stream=True
)
for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()