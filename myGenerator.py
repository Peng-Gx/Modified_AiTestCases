from langchain_core.embeddings import Embeddings
from langchain.document_loaders import WebBaseLoader
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from operator import itemgetter
from config import (
    DETECTOR_AGENT_PROMPT, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_SEARCH_K, GENERATOR_AGENT_PROMPT
)
from utils import parse_json_output

from state import TestCaseGenerationState


def detector_agent_node(prd:str, llm: BaseChatModel) -> list:
    print("--- Node: 检测器")
    if not llm:
        print("没有可用的大语言模型")
    if not prd:
        print("需求文档为空")
    try:
        app_prompt = ChatPromptTemplate.from_template(DETECTOR_AGENT_PROMPT)
        app_chain = app_prompt | llm | StrOutputParser()

        result_str = app_chain.invoke({"text": prd})
        # print(result_str)

        # 转JSON
        parsed_apps = parse_json_output(result_str, expected_type=list)

        # 文档解析失败
        if parsed_apps is None:
            # 启发式解析，判断是否返回了一个逗号分割的简单字符串
            if result_str and not result_str.startswith("I cannot") and len(
                    result_str) < 200 and '[' not in result_str and '{' not in result_str:
                possible_apps = [app.strip().strip("'\"") for app in result_str.split(',') if app.strip()]
                if possible_apps:
                    return sorted(list(set(possible_apps)))
            return []
        return parsed_apps
    except Exception as e:
        print(f"!!!!!!!!!! 在detector_agent_node中发生严重错误 !!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        return []

def getVectorstoreFromText(context:str, embeddings: Embeddings) -> FAISS:
    print("--- Node: 文本向量化")
    if not embeddings:
        print("没有可用的嵌入")
    if not context:
        print("文本为空")

    # 文本向量化
    vectorstore = None
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_text(context)
        if not splits:
            print("文本分割失败")
            return []
        vectorstore = FAISS.from_texts(splits, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"!!!!!!!!!! 文本向量化过程中发生错误 !!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        return None
    
def getVectorstoreFromUrl(url:str, embeddings: Embeddings) -> FAISS:
    print("--- Node: 文本向量化")
    if not embeddings:
        print("没有可用的嵌入")
    if not url:
        print("url为空")

    vectorstore=None
    try:
        # 1. 抓取网页
        loader = WebBaseLoader(url)
        documents = loader.load()

        # 2. 分块
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = splitter.split_documents(documents)

        # 3. 向量化并存储到 FAISS
        vectorstore = FAISS.from_documents(splits, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"!!!!!!!!!! 网页向量化过程中发生错误 !!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        return None


#保证FAISS携带embedding的引用
def addVectorstoreFromUrl(vectorstore:FAISS, url:str) -> FAISS:
    print("--- Node: 文本向量化")
    if not url:
        print("url为空")

    try:
        # 1. 抓取网页
        loader = WebBaseLoader(url)
        documents = loader.load()

        # 2. 分块
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = splitter.split_documents(documents)

        # 3. 向量化并存储到 FAISS
        vectorstore.add_documents(splits)
        return vectorstore
    except Exception as e:
        print(f"!!!!!!!!!! 网页向量化过程中发生错误 !!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        return None

def generator_agent_node(testPoint: list, llm: BaseChatModel, vectorstore: FAISS) -> list:
    print("--- Node: 生成器")
    if not llm:
        print("没有可用的大语言模型")
    if not testPoint or testPoint == []:
        print("未检测到测试点")
    if not vectorstore:
        print("未检测到文本向量数据库")

    # RAG
    retrieval_chain = None
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})
        case_generation_prompt = ChatPromptTemplate.from_template(GENERATOR_AGENT_PROMPT)
        document_chain = create_stuff_documents_chain(llm, case_generation_prompt)
        retriever_chain = create_retrieval_chain(retriever, document_chain) | itemgetter("answer")
    except Exception as e:
        print(f"!!!!!!!!!! 构建过程链过程中发生错误 !!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        return []

    if not retriever_chain:
        print("链条创建失败")
        return []
    try:
        input = str(testPoint)
        result_str = retriever_chain.invoke({"input": input})
        # print(result_str)

        # 转JSON
        parsed_case = parse_json_output(result_str, expected_type=list)

        # 文档解析失败
        if parsed_case is None:
            print("JSON数据解析失败")
            return []
        return parsed_case
    except Exception as e:
        print(f"!!!!!!!!!! 在generator_agent_node中发生严重错误 !!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        return []




