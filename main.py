import state
import llm_initial
# import Generator
# import myGenerator
from typing import TypedDict, List, Dict
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

def evaluateTotalDetectedNode(prd: str, detectedNode: List[Dict], 
                              knowledgeStore: FAISS, llm: BaseChatModel, prompt: str="") -> Dict:
    
    return None

def initFaiss(embeddingFunction: Embeddings) -> FAISS:
    faiss = FAISS.from_texts(["hello"],embedding=embeddingFunction)
    return faiss

# def addToFaissFromUrls(store: FAISS, urls: List[str], chunkSize: int, chunkOverlap: int) -> FAISS:
#     for url in urls:
#         # 1. 抓取网页
#         loader = WebBaseLoader(url)
#         documents = loader.load()

#         # 2. 分块
#         splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
#         splits = splitter.split_documents(documents)

#         # 3. 向量化并存储到 FAISS
#         store.add_documents(splits)
#     return store


# def addToFaissFromTexts(store: FAISS, texts: List[str], chunkSize: int, chunkOverlap: int) -> FAISS:
#     for text in texts:
#         # 1. 分块
#         splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
#         splits = splitter.split_text(text)
#         for s in splits:
#             print(s)
#             print("-------------------")

#         # 2. 向量化并存储到 FAISS
#         store.add_texts(splits)
#     return store

def addToFaissFromUrlsWithSemanticSplit(store: FAISS, urls: List[str], llm: BaseChatModel) -> FAISS:
    for url in urls:
        # 1. 抓取网页
        loader = WebBaseLoader(url)
        documents = loader.load()

        # 2. 分块
        splits = []
        for doc in documents:
            splits.extend(semanticSplitTextWithLLM(doc.page_content, llm))

        # 3. 向量化并存储到 FAISS
        store.add_texts(splits)
    return store

def addToFaissFromTextsWithSemanticSplit(store: FAISS, texts: List[str], llm: BaseChatModel) -> FAISS:
    for text in texts:
        # 1. 分块
        splits = semanticSplitTextWithLLM(text, llm)

        # 2. 向量化并存储到 FAISS
        store.add_texts(splits)
    return store

def semanticSplitTextWithLLM(text: str, llm: BaseChatModel, prompt: str="") -> List[str]:
    # 清洗输入文本（去除多余空行，统一换行符）
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # 合并多空行
    text = text.strip()

    if prompt=="":
        prompt = """
    你是一个专业的文档结构分析助手。

    请将以下文本按语义划分为多个段落。注意：
    - 不要划分得太碎，尽量保留完整的语义上下文；
    - 每段尽可能自洽、表达一个完整的意思；
    - 连续句子如果语义上关联紧密，不要拆开；
    - 使用三个短横线 `---` 作为不同段的分隔符；
    - 输出时不要添加任何多余的解释或标注。

    以下是需要处理的文本：

    {text}
    """
    
    app_chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
    result_str = app_chain.invoke({"text": text})
    splits = re.split(r'\n?-{3,}\n?', result_str)
    return splits



if __name__=="__main__":
    text = """
    5.1 功能概述
    5.1.1 业务流程图
    5.1.2 功能架构
    5.2 功能说明
    5.2.1 账号登录
    5.2.1.1 业务流程
    用户打开登录网址进入登录页面；
    输入登录账号、密码、验证码进行登录；
    5.2.1.2 页面原型
    5.2.1.3 逻辑说明
    账号输入框：输入账号后显示清空操作按钮，点击清空按钮清空已输入的账号；
    密码输入框：输入密码后显示密码明文查看按钮及清空按钮，鼠标按下明文查看按钮密码明文展示，鼠标松开后密文展示，点击清空按钮清空已输入密码；
    验证码：验证码为4位随机的大写英文字母，单击验证码展示区域刷新验证码，验证码有效期为60秒；
    所有输入框获取焦点时提示文字消失且边框变为绿色，失去焦点时恢复原状；
    【登录】按钮：点击【登录】按钮时依次校验登录账号、密码、验证码，若某一项校验出问题则在对应输入框下方红字提醒用户错误原因，全部校验通过则成功登录并进入系统首页；
    【找回密码】按钮：点击【找回密码】按钮跳转至找回密码页面；
    账号首次成功登录时弹窗提示账号首次登录时需修改初始密码，点击【确定】按钮跳转至修改初始密码页面，初始密码修改后需要重新输入账号、密码、验证码进行登录；
    5.2.2 重置初始密码
    5.2.2.1 业务流程
    用户初次成功登录账号后弹窗提示用户修改初始密码；
    点击弹窗的【确定】按钮后页面跳转至重置初始密码页面；
    5.2.2.2 页面原型
    5.2.2.3 业务逻辑
    新密码输入框：输入密码后显示密码明文查看按钮及清空按钮，鼠标按下明文查看按钮密码明文展示，鼠标松开后密文展示，点击清空按钮清空已输入密码；
    确认密码：输入密码后显示密码明文查看按钮及清空按钮，鼠标按下明文查看按钮密码明文展示，鼠标松开后密文展示，点击清空按钮清空已输入密码；
    验证码：点击【获取验证码】按钮向登录账号绑定手机号发送6位随机数字验证码，验证码已发送后【获取验证码】按钮展示60秒倒计时，倒计时结束后恢复原状，验证码有效期为60秒；
    所有输入框获取焦点时提示文字消失且输入框边框变为绿色，失去焦点时恢复原状；
    【取消】按钮：点击【取消】按钮取消修改初始密码，返回至登录页面；
    【提交】按钮：点击【提交】按钮依次校验密码、验证码，若某一项校验出问题则在对应输入框下方红字提醒用户错误原因，全部校验通过则提示重置密码成功，然后返回至登录页面；
    5.2.3 找回密码
    5.2.3.1 业务流程
    用户打开登录网址进入登录页面；
    忘记登录密码后在登录页面页面点击【找回密码】按钮进入找回密码页面；
    5.2.3.2 页面原型
    5.2.3.3 业务逻辑
    新密码输入框：输入密码后显示密码明文查看按钮及清空按钮，鼠标按下明文查看按钮密码明文展示，鼠标松开后密文展示，点击清空按钮清空已输入密码；
    确认密码：输入密码后显示密码明文查看按钮及清空按钮，鼠标按下明文查看按钮密码明文展示，鼠标松开后密文展示，点击清空按钮清空已输入密码；
    验证码：点击【获取验证码】按钮向登录账号绑定手机号发送6位随机数字验证码，验证码已发送后【获取验证码】按钮展示60秒倒计时，倒计时结束后恢复原状，验证码有效期为60秒；
    所有输入框获取焦点时提示文字消失且输入框边框变为绿色，失去焦点时恢复原状；
    【取消】按钮：点击【取消】按钮取消修改初始密码，返回至登录页面；
    【提交】按钮：点击【提交】按钮依次校验密码、验证码，若某一项校验出问题则在对应输入框下方红字提醒用户错误原因，全部校验通过则提示重置密码成功，然后返回至登录页面；
    5.2.4 修改密码
    5.2.4.1 业务流程
    用户成功登录账号；
    在我的页面选择修改密码进入修改密码页面；
    5.2.4.2 页面原型
    5.2.4.3 业务逻辑
    原密码输入框：输入密码后显示密码明文查看按钮及清空按钮，鼠标按下明文查看按钮密码明文展示，鼠标松开后密文展示，点击清空按钮清空已输入密码；
    新密码输入框：输入密码后显示密码明文查看按钮及清空按钮，鼠标按下明文查看按钮密码明文展示，鼠标松开后密文展示，点击清空按钮清空已输入密码；
    确认密码：输入密码后显示密码明文查看按钮及清空按钮，鼠标按下明文查看按钮密码明文展示，鼠标松开后密文展示，点击清空按钮清空已输入密码；
    验证码：点击【获取验证码】按钮向登录账号绑定手机号发送6位随机数字验证码，验证码已发送后【获取验证码】按钮展示60秒倒计时，倒计时结束后恢复原状，验证码有效期为60秒；
    所有输入框获取焦点时提示文字消失且输入框边框变为绿色，失去焦点时恢复原状；
    【取消】按钮：点击【取消】按钮取消修改密码，返回至我的页面；
    【提交】按钮：点击【提交】按钮依次校验密码、验证码，若某一项校验出问题则在对应输入框下方红字提醒用户错误原因，全部校验通过则提示重置密码成功；
    """

    llm, embeddings = llm_initial.initialize_llm()

    # sta = state.TestCaseGenerationState(prd_content=text, prd_vector=None, detected_testPoint=[], generated_cases=[], evaluation_report={}, optimization_hints=[])

    # detected = myGenerator.detector_agent_node(sta["prd_content"], llm)
    # sta["detected_testPoint"].append(detected)

    # vectorstore = myGenerator.getVectorstore(sta["prd_content"], embeddings)
    # sta["prd_vector"]=vectorstore


    # clip=20
    # for i in range(len(detected)//clip+1):
    #     begin=clip*i
    #     if(begin+clip>len(detected)):
    #         end=len(detected)
    #     else:
    #         end=begin+clip
    #     generated = myGenerator.generator_agent_node(detected[begin:end], llm, vectorstore)
    #     sta["generated_cases"].append(generated)


    # print(sta["detected_testPoint"])
    # print(sta["generated_cases"])

    url_0 = "https://www.cnblogs.com/darlingchen/p/16241534.html?utm_source=chatgpt.com"
    url_1 = "https://www.cnblogs.com/Uni-Hoang/p/13204907.html?utm_source=chatgpt.com"

    faiss = initFaiss(embeddings)
    faiss = addToFaissFromUrlsWithSemanticSplit(faiss, [url_0,url_1], llm)

    # faiss_text = initFaiss(embeddings)
    # faiss_text = addToFaissFromTexts(faiss_text, [text],300, 30)
