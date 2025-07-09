import state
import llm_initial
import Generator
from typing import TypedDict, List, Dict
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from utils import parse_json_output
import logging

def generateDetectedNode(prd: str, llm: BaseChatModel, prompt: str=None) -> List[Dict]:
    if not prompt:
        prompt="""
# 角色：你是一位经验丰富的质量保证测试分析师，具备出色的文档分析能力和系统拆解能力。
# 任务：你的核心任务是接收用户给你的需求文档，仔仔细细的阅读，然后准确识别其中涉及到的每个功能以及存在的测试点。
# 核心指令：
1. 仔细阅读并理解下方输入信息中提供的需求文档信息。
2. 从该需求文档中分析出该文档涉及到的所有的系统功能。
3. 对于你识别到的每一项系统功能，分析出其中可能的每一项测试点。
# 输出格式要求：
* 你的回复**必须且只能是**一个Python风格的列表List，这个列表里面的每个元素都是一个Python风格的字典Dict，这个字典里面包含以下三个键，其中, "case_ID"为该测试案例的唯一标识，自增，"function"和"testPoint", 其中"function"包含你分析出来的功能，"testPoint"为你从该功能中分析出来的单个测试点，三个键都是字符串。
* **绝对不要**在列表的 `[` 之前或 `]` 之后添加任何介绍、解释、注释、道歉或其他任何文字。
* 如果需求文档中没有明确的系统功能或测试点，**必须**返回一个空列表：`[]`。
* 注意字典中的键是字符串，要用""包裹。
# 输出示例：[
    {{"case_ID": "1", "function": "登录功能", "testPoint": "用户密码错误"}},
    {{"case_ID": "2", "function": "登录功能", "testPoint": "用户密码长度超过上限"}},
    {{"case_ID": "3", "function": "注销功能", "testPoint": "用户未登录时注销"}}
]
# 输入信息：
```{text}```
"""

    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
    result_str = chain.invoke({"text": prd})
    logging.info("生成测试点")
    logging.info(result_str)

    result_json = parse_json_output(result_str, expected_type=list)
    return result_json


def generateDetectedNodeWithEval(prd:str, detectedNode: List[Dict], eval:Dict, llm: BaseChatModel, prompt: str=None) -> List[Dict]:

    if not prompt:
        prompt="""
# 角色：
你是一位资深的软件测试设计专家，擅长根据需求文档和评估反馈，系统性地优化测试点列表，提升质量和覆盖度。

# 任务目标：
你将收到以下三部分输入：
1. 原始测试点列表（基于 PRD 初步生成）
2. 对这些测试点的评估报告（包括评分、问题分析和修改建议，可能参考了知识库）
3. 原始需求文档（PRD）

你的任务是：
- 基于原始测试点列表，结合评估报告中的问题分析和修改建议，以及需求文档的真实需求，**对测试点列表进行增删改操作**。
- 确保所有测试点均有真实需求支持，删除无依据的幻觉项。
- 修复测试点中深度不足、异常场景遗漏、边界值未覆盖、冗余或表述模糊的问题，且尽量保持列表结构和逻辑完整。
- 不要完全重写或从零生成新的测试点列表，而是对现有列表做精准的优化。

# 输出格式要求：
* 你的回复**必须且只能是**一个Python风格的列表 `List`，其中每个元素是一个Python风格的字典 `Dict`。
* 每个字典包含以下三个键：
  - `"case_ID"`：测试点唯一标识，**基于原列表，从 "1" 开始递增**，保持连续字符串编号。
  - `"function"`：功能模块名称（可依据需求文档和评估报告调整或细化）。
  - `"testPoint"`：优化后的单个测试点描述。
* **绝对不要**在列表的 `[` 前或 `]` 后添加任何文字、解释、注释或道歉。
* 如果需求文档中无明确功能或测试点，返回空列表：`[]`。
* 所有键名均需用双引号 `"` 包裹，确保合法 Python 字典格式。

# 输出示例：
[
    {{"case_ID": "1", "function": "登录功能", "testPoint": "用户密码错误"}},
    {{"case_ID": "2", "function": "登录功能", "testPoint": "用户密码长度超过上限"}},
    {{"case_ID": "3", "function": "注销功能", "testPoint": "用户未登录时注销"}}
]

# 输入部分：

## 原始测试点列表：
```{testPoints}```
## 评估报告（含修改建议和知识支持）：
```{evaluation}```
## 原始需求文档（PRD）：
```{document}```
"""
    nodes_text = "\n\n".join([
        "\n".join([f"{k}: {v}" for k, v in node.items()])
        for node in detectedNode
    ])

    eval_text = "\n".join([f"{k}: {v}" for k,v in eval.items()])

    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
    result_str = chain.invoke({"testPoints": nodes_text, "evaluation": eval_text, "document": prd})
    logging.info("重新生成测试点")
    logging.info(result_str)

    result_json =  parse_json_output(result_str, expected_type=list)

    return result_json

def evaluateTotalDetectedNode(prd: str, detectedNode: List[Dict], 
                              knowledgeStore: FAISS, llm: BaseChatModel, prompt: str=None) -> Dict:
    prd_splits = semanticSplitTextWithLLM(prd, llm)

    # 1. 构建检索 query（合并字典值）
    queries = prd_splits + [" ".join(node.values()) for node in detectedNode]

    # 2. 检索 + 去重
    retrieved_docs = []
    for query in queries:
        docs = knowledgeStore.similarity_search(query, k=4)
        retrieved_docs.extend(docs)

    seen = set()
    unique_contents = []
    for doc in retrieved_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_contents.append(doc.page_content)

    context = "\n".join(unique_contents)

    # 3. 格式化 detectedNode 内容
    nodes_text = "\n\n".join([
        "\n".join([f"{k}: {v}" for k, v in node.items()])
        for node in detectedNode
    ])

    # 4. Prompt 模板
    if not prompt:
        prompt = """
# 角色：
你是一位经验丰富的软件测试架构师，具备卓越的需求分析和测试用例评估能力，注重细节，追求覆盖率和测试质量的极致提升。

# 任务：
你的核心任务是基于下方提供的需求文档、支持信息（知识库）以及测试点列表，依据知识库中定义的评估准则，对测试点列表进行全面评估与优化。

# 核心指令：
1. 详读需求文档、支持信息和测试点列表。
2. 依据支持信息中的评估准则，严格审视测试点列表在测试深度和相关性两个维度上的表现。
3. 按照知识库中评估准则的标准，对每个维度分别打分（1-5分）：
   - 5分：完美无缺，达到世界级标准
   - 4分：优秀，仅有极少改进空间
   - 3分：存在明显不足
   - 2分：存在严重缺陷
   - 1分：完全未覆盖该维度
4. 根据两个维度得分及预设权重（测试深度占70%，相关性占30%），计算加权综合分（四舍五入至整数，返回字符串格式）。
5. 输出包含三个键的 Python 字典：
   - "score"：加权综合分（字符串）
   - "evaluation"：对各维度评分的说明，以及对应的改进建议和优化方案，合并为一段文字，避免冗余。
   - "recommendations"：针对测试点列表的具体修改建议，包含需新增、删除或改写的测试点及理由。

# 输出格式要求：
* 仅输出一个 Python 字典，且仅包含 "score"、"evaluation"、"recommendations" 三个键。
* 不允许包含其他内容或额外说明。

# 输入内容：
- 需求文档：
```{document}```
- 支持信息（知识库检索内容，含评估准则）：
```{context}```
- 测试点列表：
```{testPoint}```
"""

    # 5. 构造 chain
    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
    result_str = chain.invoke({"document":prd,"context":context, "testPoint":nodes_text})
    logging.info("评估测试点")
    logging.info(result_str)

    result_json = parse_json_output(result_str, expected_type=dict)
    return result_json

def initFaiss(embeddingFunction: Embeddings) -> FAISS:
    faiss = FAISS.from_texts(["hello"],embedding=embeddingFunction)
    return faiss

def addToFaissFromUrls(store: FAISS, urls: List[str], chunkSize: int, chunkOverlap: int) -> FAISS:
    for url in urls:
        # 1. 抓取网页
        loader = WebBaseLoader(url)
        documents = loader.load()

        # 2. 分块
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
        splits = splitter.split_documents(documents)

        # 3. 向量化并存储到 FAISS
        store.add_documents(splits)
    return store


def addToFaissFromTexts(store: FAISS, texts: List[str], chunkSize: int, chunkOverlap: int) -> FAISS:
    for text in texts:
        # 1. 分块
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
        splits = splitter.split_text(text)

        # 2. 向量化并存储到 FAISS
        store.add_texts(splits)
    return store

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

def semanticSplitTextWithLLM(text: str, llm: BaseChatModel, prompt: str=None) -> List[str]:
    # 清洗输入文本（去除多余空行，统一换行符）
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # 合并多空行
    text = text.strip()

    if not prompt:
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
    
    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
    result_str = chain.invoke({"text": text})
    splits = re.split(r'\n?-{3,}\n?', result_str)
    return splits



if __name__=="__main__":
    logging.basicConfig(filename="logfile.txt", level=logging.INFO, encoding="utf-8")

    # 需求文档
    prd = """
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

    # 模型初始化
    llm, embeddings = llm_initial.initialize_llm()
    
    # 初次生成测试点
    detectedNode = generateDetectedNode(prd, llm)

    # 加载知识库
    url_0 = "https://www.cnblogs.com/darlingchen/p/16241534.html?utm_source=chatgpt.com"
    url_1 = "https://www.cnblogs.com/Uni-Hoang/p/13204907.html?utm_source=chatgpt.com"

    knowledgeStore = initFaiss(embeddings)
    knowledgeStore = addToFaissFromUrlsWithSemanticSplit(knowledgeStore, [url_0,url_1], llm)

    # 评估
    bestDetectedNode = detectedNode
    bestScore = 0.0

    times = 5
    for i in range(times):
        evalReport = evaluateTotalDetectedNode(prd, detectedNode, knowledgeStore, llm)
        if float(evalReport["score"]) > bestScore:
            bestScore = float(evalReport["score"])
            bestDetectedNode = detectedNode
        if i<times-1:
            detectedNode = generateDetectedNodeWithEval(prd, detectedNode, evalReport, llm)
