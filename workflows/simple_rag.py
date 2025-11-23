from dotenv import load_dotenv
load_dotenv()


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langsmith import Client
from typing_extensions import List, TypedDict

from etc import set_path
set_path()
from src.path import DB_DIR
from src.config import AIConfig


# ==================== 설정 변수 ====================
LLM_MODEL = AIConfig.MODEL_NAME
EMBEDDING_MODEL = AIConfig.EMBEDDING_MODEL  # 벡터 DB 생성 시 사용한 모델과 동일해야 함
RETRIEVER_K = 3
COLLECTION_NAME = "income_tax"


# ==================== 1. 기존 Vector DB 로드 및 Retriever 설정 ====================
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)

retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})


# ==================== 2. LangGraph RAG 구축 ====================
# Agent 상태 정의
class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str


# 문서 검색 노드
def retrieve_node(state: AgentState):
    query = state["query"]
    docs = retriever.invoke(query)
    return {"context": docs}


# 답변 생성 노드
def generate(state: AgentState):
    context = state["context"]
    query = state["query"]
    
    # LangSmith에서 RAG 프롬프트 가져오기
    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
    
    # LLM 설정
    llm = ChatOpenAI(model=LLM_MODEL)
    
    # RAG 체인 실행
    rag_chain = prompt | llm
    response = rag_chain.invoke({
        "question": query,
        "context": context,
    })
    
    return {"answer": response}


# 그래프 구성
graph_builder = StateGraph(AgentState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


# ==================== 3. 실행 예시 ====================
if __name__ == "__main__":
    initial_state = {"query": "연봉 5천만원 직장인의 소득세는?"}
    result = graph.invoke(initial_state)
    print(result)
