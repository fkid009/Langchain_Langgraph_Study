from pathlib import Path
import sys
from typing import Literal

from dotenv import load_dotenv
load_dotenv()
from typing_extensions import List, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langsmith import Client

from etc import set_path
set_path()
from src.path import DB_DIR  # DATA_DIR는 안 쓰이지만, 패턴 맞춰둠
from src.config import AIConfig


# ==================== Config ====================
LLM_MODEL = AIConfig.MODEL_NAME
EMBEDDING_MODEL = AIConfig.EMBEDDING_MODEL
COLLECTION_NAME = "income_tax"
RETRIEVER_K = 3

# rewrite 시 사용할 간단한 사전
DICTIONARY = ["사람과 관련된 표현 -> 거주자"]


# ==================== 1. Vector Store & Retriever ====================
embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
)

retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})


# ==================== 2. LangGraph State & Prompts ====================
class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str


graph_builder = StateGraph(AgentState)

client = Client()
rag_prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
rel_prompt = client.pull_prompt("langchain-ai/rag-document-relevance", include_model=True)

llm = ChatOpenAI(model=LLM_MODEL)

rewrite_prompt = PromptTemplate.from_template(
    """
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    사전: {dictionary}
    질문: {query}
    """
)


# ==================== 3. Node Functions ====================
def retrieve(state: AgentState) -> dict:
    query = state["query"]
    docs = retriever.invoke(query)
    return {"context": docs}


def generate(state: AgentState) -> dict:
    context = state["context"]
    query = state["query"]

    rag_chain = rag_prompt | llm
    response = rag_chain.invoke(
        {
            "question": query,
            "context": context,
        }
    )
    return {"answer": response}


def check_doc_relevance(state: AgentState) -> Literal["generate", "rewrite"]:
    query = state["query"]
    context = state["context"]

    rel_chain = rel_prompt | llm
    response = rel_chain.invoke(
        {
            "question": query,
            "documents": context,
        }
    )
    print(f"context: {context}")
    print(f"relevance response: {response}")

    # LangSmith 프롬프트 응답 형식에 맞게 조정 필요할 수 있음
    if response.get("Score") == 1:
        return "generate"
    return "rewrite"


def rewrite(state: AgentState) -> dict:
    query = state["query"]
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    response = rewrite_chain.invoke({"query": query, "dictionary": DICTIONARY})
    return {"query": response}


# ==================== 4. Graph Build ====================
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite", rewrite)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges("retrieve", check_doc_relevance)
graph_builder.add_edge("rewrite", "retrieve")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


# ==================== 5. Run Example ====================
if __name__ == "__main__":
    initial_state = {"query": "연봉 5천만원 세금"}
    result = graph.invoke(initial_state)
    print(result)
