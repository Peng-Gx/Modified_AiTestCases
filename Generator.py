from langchain_core.embeddings import Embeddings
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


def detector_agent_node(state:TestCaseGenerationState, llm: BaseChatModel) -> list:
    print("--- Node: 检测器")
    if not llm:
        print("没有可用的大语言模型")
    if not state or state["prd_content"] is None:
        print("需求文档为空")
    try:
        app_prompt = ChatPromptTemplate.from_template(DETECTOR_AGENT_PROMPT)
        app_chain = app_prompt | llm | StrOutputParser()

        result_str = app_chain.invoke({"text": state["prd_content"]})
        print(result_str)

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

def generator_agent_node(state:TestCaseGenerationState, llm: BaseChatModel, embeddings: Embeddings) -> list:
    print("--- Node: 生成器")
    if not llm:
        print("没有可用的大语言模型")
    if not embeddings:
        print("没有可用的嵌入")
    if not state or state["prd_content"] is None:
        print("需求文档为空")
    if not state or state["detected_testPoint"] == []:
        print("未检测到测试点")

    # 文本向量化
    vectorstore = None
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_text(state["prd_content"])
        if not splits:
            print("文本分割失败")
            return []
        vectorstore = FAISS.from_texts(splits, embedding=embeddings)
    except Exception as e:
        print(f"!!!!!!!!!! 文本向量化过程中发生错误 !!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        return []

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
        input = str(state["detected_testPoint"])
        result_str = retriever_chain.invoke({"input": input})
        print(result_str)

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




