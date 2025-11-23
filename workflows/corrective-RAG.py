from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from typing_extensions import TypedDict

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langsmith import Client
from langchain_community.tools import TavilySearchResults

from etc import set_path
set_path()
from src.path import DB_DIR
from src.config import AIConfig




# ==================== Config ====================
LLM_MODEL = AIConfig.MODEL_NAME
EMBEDDING_MODEL = AIConfig.EMBEDDING_MODEL
COLLECTION_NAME = "income_tax"
RETRIEVER_K = 3


# ==================== 1. Vector Store & Retriever ====================
embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
)

retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})


# ==================== 2. State & Graph Builder ====================
class AgentState(TypedDict):
    query: str
    context: list  # mixed type: Document list or web search result list
    answer: str


graph_builder = StateGraph(AgentState)


# ==================== 3. LLM, Prompts, Tools ====================
client = Client()
rag_prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
rel_prompt = client.pull_prompt("langchain-ai/rag-document-relevance", include_model=True)

llm = ChatOpenAI(model=LLM_MODEL)

rewrite_prompt = PromptTemplate.from_template(
    """
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    질문: {query}
    """
)

tavily_search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)


# ==================== 4. Node Definitions ====================
def retrieve(state: AgentState) -> dict:
    query = state["query"]
    docs = retriever.invoke(query)
    return {"context": docs}


def check_doc_relevance(state: AgentState) -> Literal["relevant", "irrelevant"]:
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

    if response.get("Score") == 1:
        return "relevant"
    return "irrelevant"


def rewrite(state: AgentState) -> dict:
    query = state["query"]
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    response = rewrite_chain.invoke({"query": query})
    print(f"Rewritten query: {response}")
    return {"query": response}


def generate(state: AgentState) -> dict:
    context = state["context"]
    query = state["query"]

    rag_chain = rag_prompt | llm | StrOutputParser()
    response = rag_chain.invoke(
        {
            "question": query,
            "context": context,
        }
    )
    return {"answer": response}


def web_search(state: AgentState) -> dict:
    query = state["query"]
    search_results = tavily_search_tool.invoke(query)
    print(f"Web search results: {search_results}")
    return {"context": search_results}


# ==================== 5. Graph Wiring ====================
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("web_search", web_search)

graph_builder.add_edge(START, "retrieve")

graph_builder.add_conditional_edges(
    "retrieve",
    check_doc_relevance,
    {
        "relevant": "generate",
        "irrelevant": "rewrite",
    },
)

graph_builder.add_edge("rewrite", "web_search")
graph_builder.add_edge("web_search", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


# ==================== 6. Visualize & Run Examples ====================
if __name__ == "__main__":
    # Example 1: income tax question
    initial_state = {"query": "연봉 5천만원 세금"}
    result_1 = graph.invoke(initial_state)
    print("Result 1:", result_1)

    # Example 2: general web search question
    initial_state = {"query": "역삼역 맛집을 알려줘"}
    result_2 = graph.invoke(initial_state)
    print("Result 2:", result_2)
