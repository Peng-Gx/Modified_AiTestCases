import llm_initial

llm, embeddings = llm_initial.initialize_llm()
response = llm.invoke("你好，火山引擎！")
print(response)