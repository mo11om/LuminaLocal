import os
import json
from typing import TypedDict, Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain Core / Community Imports (NO legacy 'langchain' package)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Direct cross-encoder import (no langchain wrapper)
from sentence_transformers import CrossEncoder

# ==========================================
# 1. Configuration (Shared)
# ==========================================
class Config:
    # Resolve the project root (2 levels up from src/cv_matcher/utils.py)
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _config_path = os.path.join(_project_root, "config.json")
    
    if os.path.exists(_config_path):
        with open(_config_path, "r") as f:
            _data = json.load(f)
    else:
        # Fallback default values or raise error
        print(f"⚠️ Config file not found at {_config_path}, using defaults")
        _data = {}

    # === ADD THE PROVIDER HERE ===
    LLM_PROVIDER = _data.get("LLM_PROVIDER", "ollama")
    LLM_MODEL = _data.get("LLM_MODEL", "gpt-oss:20b")
 
    EMBEDDING_MODEL = _data.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_DB_PATH = _data.get("VECTOR_DB_PATH", "./data/chroma_db")
    CHUNK_SIZE = _data.get("CHUNK_SIZE", 500)
    CHUNK_OVERLAP = _data.get("CHUNK_OVERLAP", 50)
    # Paths for specific scripts
    JOBS_FILE = _data.get("JOBS_FILE", "./data/jobs/processed_jobs_schema.json")
    RESUME_FILE = _data.get("RESUME_FILE", "./data/raw/CV.md.pdf")
    OUTPUT_DIR = _data.get("OUTPUT_DIR", "./data/output")
    
    # Hyperparameters
    RETRIEVER_K = _data.get("RETRIEVER_K", 8)
    ANALYSIS_OUTPUT_CSV = _data.get("ANALYSIS_OUTPUT_CSV", "analysis_results.csv")
    
    # GPT Baseline Configuration
    GPT_MODEL = _data.get("GPT_MODEL", "gpt-4o")
    GPT_TEMPERATURE = _data.get("GPT_TEMPERATURE", 0.3)
    GPT_MAX_TOKENS = _data.get("GPT_MAX_TOKENS", 500)
    ENABLE_GPT_BASELINE = _data.get("ENABLE_GPT_BASELINE", True)

    # --- NEW: Re-ranker Configuration (Step 3) ---
    RERANKER_MODEL = _data.get("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    RERANK_TOP_N = _data.get("RERANK_TOP_N", 8)

# ==========================================
# 2. Shared Types
# ==========================================
class BaseAgentState(TypedDict):
    """
    Base state for the graph. Individual scripts can extend this 
    or just use dynamic keys if they need extra fields like 'job_info'.
    """
    job_description: str        # Flattened string for retrieval query
    context: str                # Retrieved resume chunks (labeled)
    analysis: str               # Final JSON string output
    job_info: Optional[Dict[str, Any]]  # Optional: for detailed job objects
    optimized_queries: List[str]        # NEW (Step 1): Decomposed sub-queries

# ==========================================
# 3. Resume Ingestion Logic
# ==========================================
class ResumeIngestor:
    def __init__(self):
        print(f"🔄 Initializing Embedding Model: {Config.EMBEDDING_MODEL}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            # model_kwargs={'device': 'hip'} # Uncomment if you have a GPU
                                                )
        self.vector_store = None
        self.chunks = None  # NEW: Store raw Document objects for BM25

    def get_vector_store(self, pdf_path: str, force_reload: bool = False):
        """
        Loads the PDF and creates/returns the Vector Store.
        Also persists raw document chunks for BM25Retriever.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Resume file not found: {pdf_path}")

        print(f"📄 Loading Resume: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        self.chunks = texts  # NEW: Persist for BM25Retriever
        print(f"🧩 Split resume into {len(texts)} chunks.")

        print("💾 Creating/Updating Vector Store...")
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=Config.VECTOR_DB_PATH
        )
        print("✅ Vector Store Ready!")
        return self.vector_store

    def get_chunks(self):
        """Return raw document chunks for BM25Retriever initialization."""
        if self.chunks is None:
            raise ValueError("No chunks available. Call get_vector_store() first.")
        return self.chunks

# ==========================================
# 4. Hybrid Retriever (Steps 2 & 3)
#    Uses only langchain-core + langchain-community + sentence-transformers.
#    NO dependency on the legacy 'langchain' package.
# ==========================================
class HybridRetriever(BaseRetriever):
    """
    Custom hybrid retriever pipeline:
      Step 2: BM25 (lexical) + Vector (semantic) via Reciprocal Rank Fusion
      Step 3: Cross-encoder re-ranking via sentence-transformers
    """
    vector_retriever: Any = None
    bm25_retriever: Any = None
    cross_encoder_model: Any = None
    top_n: int = 8
    vector_weight: float = 0.5
    bm25_weight: float = 0.5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # --- Step 2: Ensemble via Reciprocal Rank Fusion ---
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        fused = self._reciprocal_rank_fusion(vector_docs, bm25_docs)

        # --- Step 3: Cross-encoder re-ranking ---
        if self.cross_encoder_model and fused:
            pairs = [(query, doc.page_content) for doc in fused]
            scores = self.cross_encoder_model.predict(pairs)
            scored_docs = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[: self.top_n]]

        return fused[: self.top_n]

    def _reciprocal_rank_fusion(
        self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60
    ) -> List[Document]:
        """Reciprocal Rank Fusion (RRF) of two ranked lists."""
        doc_scores: Dict[int, float] = {}
        doc_map: Dict[int, Document] = {}

        for rank, doc in enumerate(vector_docs):
            key = hash(doc.page_content)
            doc_scores[key] = doc_scores.get(key, 0) + self.vector_weight * (
                1 / (k + rank + 1)
            )
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = hash(doc.page_content)
            doc_scores[key] = doc_scores.get(key, 0) + self.bm25_weight * (
                1 / (k + rank + 1)
            )
            doc_map[key] = doc

        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys]


def build_hybrid_retriever(
    vector_store, chunks, retriever_k: int = None, rerank_top_n: int = None
):
    """
    Build a production hybrid retriever pipeline:
      Step 2: BM25 (lexical) + Vector (semantic) → RRF Ensemble
      Step 3: Cross-encoder re-ranking → top_n selection

    Uses only langchain-core, langchain-community, and sentence-transformers.

    Args:
        vector_store: Chroma vector store instance.
        chunks: List of Document objects (same chunks used by vector store).
        retriever_k: Number of docs each base retriever returns. Defaults to Config.RETRIEVER_K.
        rerank_top_n: Number of docs the re-ranker keeps. Defaults to Config.RERANK_TOP_N.

    Returns:
        HybridRetriever instance (subclass of BaseRetriever).
    """
    k = retriever_k or Config.RETRIEVER_K
    top_n = rerank_top_n or Config.RERANK_TOP_N

    # --- Step 2a: Semantic vector retriever ---
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # --- Step 2b: BM25 lexical retriever from the same chunks ---
    # TODO: V2 Patch - Inject custom jieba tokenizer preprocess_func here for better Chinese soft-skill lexical matching.
    bm25_retriever = BM25Retriever.from_documents(chunks, k=k)
    print(f"✅ EnsembleRetriever ready (Vector 0.5 + BM25 0.5, k={k})")

    # --- Step 3: Load cross-encoder for re-ranking ---
    print(f"🔄 Loading re-ranker model: {Config.RERANKER_MODEL}...")
    cross_encoder = CrossEncoder(
        Config.RERANKER_MODEL,
        trust_remote_code=True,
    )
    print(f"✅ Hybrid Retriever (RRF Ensemble + Cross-Encoder Re-ranking, top_n={top_n}) Ready!")

    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        cross_encoder_model=cross_encoder,
        top_n=top_n,
    )