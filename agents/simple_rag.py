"""
소득세법 문서 기반 Simple RAG (Retrieval-Augmented Generation) 시스템
"""

# ==================== Import ====================
import asyncio
import markdown
from bs4 import BeautifulSoup
import nest_asyncio
from dotenv import load_dotenv
from pyzerox import zerox

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langsmith import Client
from typing_extensions import List, TypedDict


# ==================== 환경 설정 ====================
load_dotenv()
nest_asyncio.apply()

from etc import set_path
set_path()
from src.path import DATA_DIR


# ==================== 설정 변수 ====================
FNAME = "소득세법"
MD_MODEL = "gpt-4o-mini"
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
RETRIEVER_K = 3


# ==================== 1. PDF를 Markdown으로 변환 ====================
async def main():
    file_path = str(DATA_DIR / f"{FNAME}.pdf")
    select_pages = None  # None for all pages
    output_dir = str(DATA_DIR)
    
    result = await zerox(
        file_path=file_path,
        model=MD_MODEL,
        output_dir=output_dir,
        select_pages=select_pages
    )
    return result

result = asyncio.run(main())


# ==================== 2. Markdown을 Text로 변환 ====================
# Markdown 파일 읽기 (인코딩 처리)
try:
    with open(DATA_DIR / f"{FNAME}.md", "r", encoding="cp949") as f:
        md_content = f.read()
except:
    with open(DATA_DIR / f"{FNAME}.md", "r", encoding="utf-8") as f:
        md_content = f.read()

# Markdown -> HTML -> Text 변환
html_content = markdown.markdown(md_content)
soup = BeautifulSoup(html_content, "html.parser")
text_content = soup.get_text()

# Text 파일 저장
with open(DATA_DIR / f"{FNAME}.txt", "w", encoding="utf-8") as f:
    f.write(text_content)


# ==================== 3. 텍스트 분할 및 Vector DB 구축 ====================
# 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n"]
)

# 문서 로드 및 분할
loader = TextLoader(DATA_DIR / f"{FNAME}.txt", encoding="utf-8")
document_list = loader.load_and_split(text_splitter)

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Vector DB 생성
vector_db = Chroma.from_documents(
    documents=document_list,
    embedding=embeddings,
    collection_name="income_tax",
    persist_directory=str(DATA_DIR / "db")
)

# Retriever 설정
retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})


# ==================== 4. LangGraph RAG 구축 ====================
# Agent 상태 정의
class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str


# 문서 검색 노드
def retrieve(state: AgentState):
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
        "context": context
    })
    
    return {"answer": response}


# 그래프 구성
graph_builder = StateGraph(AgentState)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


# ==================== 5. 실행 ====================
initial_state = {"query": "연봉 5천만원 직장인의 소득세는?"}
result = graph.invoke(initial_state)

print(result)