import asyncio
import nest_asyncio

from dotenv import load_dotenv
from pyzerox import zerox
import markdown
from bs4 import BeautifulSoup

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ==================== Environment Setup ====================
load_dotenv()
nest_asyncio.apply()

from etc import set_path
set_path()
from src.path import DATA_DIR, DB_DIR
from src.config import AIConfig
from src.utils import load_md, save_txt


# ==================== Config Values ====================
FNAME = "소득세법"
LLM_MODEL = AIConfig.MODEL_NAME
EMBEDDING_MODEL = AIConfig.EMBEDDING_MODEL
CHUNK_SIZE = AIConfig.CHUNK_SIZE
CHUNK_OVERLAP = AIConfig.CHUNK_OVERLAP


# ==================== 1. PDF -> Markdown ====================
async def _extract_pdf_to_md(fname: str = FNAME) -> dict:
    """Convert PDF to Markdown using Zerox."""
    file_path = str(DATA_DIR / f"{fname}.pdf")
    output_dir = str(DATA_DIR)

    print("Step 1: Converting PDF to Markdown...")
    result = await zerox(
        file_path=file_path,
        model=LLM_MODEL,
        output_dir=output_dir,
        select_pages=None,
    )
    print("PDF to Markdown conversion completed.")
    return result


def extract_pdf_to_md(fname: str = FNAME) -> dict:
    """Synchronous wrapper."""
    return asyncio.run(_extract_pdf_to_md(fname))


# ==================== 2. Markdown -> Text ====================
def convert_md_to_txt(fname: str = FNAME) -> str:
    """Convert Markdown file to plain text."""
    print("Step 2: Converting Markdown to plain text...")

    md_path = DATA_DIR / f"{fname}.md"
    txt_path = DATA_DIR / f"{fname}.txt"

    md_content = load_md(md_path)

    html_content = markdown.markdown(md_content)
    soup = BeautifulSoup(html_content, "html.parser")
    text_content = soup.get_text()

    save_txt(txt_path, text_content)

    print("Markdown to text conversion completed.")
    return str(txt_path)


# ==================== 3. Split Text & Build Vector DB ====================
def build_vector_db_from_txt(
    txt_path: str,
    collection_name: str = "income_tax",
) -> Chroma:
    """Split text and build a Chroma vector database."""
    print("Step 3: Creating text chunks and building vector database...")

    DB_DIR.mkdir(exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n"],
    )

    loader = TextLoader(txt_path, encoding="utf-8")
    document_list = loader.load_and_split(text_splitter)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=document_list,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=DB_DIR,
    )

    print("Vector database was successfully created and saved.")
    return vector_db


# ==================== 4. Full Pipeline ====================
def main(
    fname: str = FNAME,
    collection_name: str = "income_tax",
) -> Chroma:
    """Run the full pipeline: PDF -> MD -> TXT -> Vector DB."""
    print("Starting the vector database build process...")

    extract_pdf_to_md(fname=fname)
    txt_path = convert_md_to_txt(fname=fname)

    vector_db = build_vector_db_from_txt(
        txt_path=txt_path,
        collection_name=collection_name,
    )

    print("Process completed.")
    return vector_db


# ==================== 5. Execution Entry ====================
if __name__ == "__main__":
    db = main()
    print("Vector DB for income tax has been built and persisted.")
