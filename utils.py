import os
import json
from typing import TypedDict, Optional, Dict, Any

# LangChain / Community Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. Configuration (Shared)
# ==========================================
class Config:
    _config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config.json")
    
    if os.path.exists(_config_path):
        with open(_config_path, "r") as f:
            _data = json.load(f)
    else:
        # Fallback default values or raise error
        print(f"⚠️ Config file not found at {_config_path}, using defaults")
        _data = {}

    LLM_MODEL = _data.get("LLM_MODEL", "gpt-oss:20b") 
    EMBEDDING_MODEL = _data.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_DB_PATH = _data.get("VECTOR_DB_PATH", "./chroma_db")
    CHUNK_SIZE = _data.get("CHUNK_SIZE", 500)
    CHUNK_OVERLAP = _data.get("CHUNK_OVERLAP", 50)
    # Paths for specific scripts
    JOBS_FILE = _data.get("JOBS_FILE", "./jobs/processed_jobs_schema_all.json")
    RESUME_FILE = _data.get("RESUME_FILE", "./cv/CV.md.pdf")
    OUTPUT_DIR = _data.get("OUTPUT_DIR", "./output")
    
    # Hyperparameters
    RETRIEVER_K = _data.get("RETRIEVER_K", 8)
    ANALYSIS_OUTPUT_CSV = _data.get("ANALYSIS_OUTPUT_CSV", "analysis_results.csv")
# ==========================================
# 2. Shared Types
# ==========================================
class BaseAgentState(TypedDict):
    """
    Base state for the graph. Individual scripts can extend this 
    or just use dynamic keys if they need extra fields like 'job_info'.
    """
    job_description: str  # Flattened string for retrieval query
    context: str          # Retrieved resume chunks
    analysis: str         # Final JSON string output
    job_info: Optional[Dict[str, Any]] # Optional: for detailed job objects

# ==========================================
# 3. Resume Ingestion Logic
# ==========================================
class ResumeIngestor:
    def __init__(self):
        print(f"🔄 Initializing Embedding Model: {Config.EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vector_store = None

    def get_vector_store(self, pdf_path: str, force_reload: bool = False):
        """
        Loads the PDF and creates/returns the Vector Store.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Resume file not found: {pdf_path}")

        # Optional: Logic to skip processing if DB exists could go here.
        # For now, we follow the original logic of processing on run.
        
        print(f"📄 Loading Resume: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        print(f"🧩 Split resume into {len(texts)} chunks.")

        print("💾 Creating/Updating Vector Store...")
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=Config.VECTOR_DB_PATH
        )
        print("✅ Vector Store Ready!")
        return self.vector_store