from typing import Tuple, Optional
from langchain_core.language_models.chat_models import  BaseChatModel
from langchain_core.embeddings import Embeddings
from utils import import_class
from doubaoEmbedding import DoubaoEmbedding
from config import (
    VOLCENGINE_API_SECRET, LLM_PROVIDER_CONFIG, VOLCENGINE_API_BASE_URL, EMBEDDINGS_API_BASE_URL
)

def initialize_llm() -> Tuple[Optional[BaseChatModel], Optional[Embeddings]]:
    # 凭证
    api_key = VOLCENGINE_API_SECRET
    # 模型
    LLM_Class = import_class(LLM_PROVIDER_CONFIG["Volcengine"]["llm_module"], LLM_PROVIDER_CONFIG["Volcengine"]["llm_class"])
    if not LLM_Class:
        print("初始化模型类")
        return None, None
    try:
        llm = LLM_Class(openai_api_key=VOLCENGINE_API_SECRET,
                        openai_api_base=VOLCENGINE_API_BASE_URL,
                        model_name=LLM_PROVIDER_CONFIG["Volcengine"]["model_endpoint"]["doubao"],
                        max_tokens=512)
        embeddings = DoubaoEmbedding(api_url=EMBEDDINGS_API_BASE_URL,
                                     api_key=VOLCENGINE_API_SECRET,
                                     model_name="doubao-embedding-large-text-250515")
        return llm, embeddings
    except Exception as e:
        print(e)
        return None, None





