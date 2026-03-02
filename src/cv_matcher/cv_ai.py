import os
from typing import List, Dict, TypedDict

# ==========================================
# 1. 新版 Import (LangGraph & LangChain v0.2+)
# ==========================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # 修正後的路徑
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # 建議使用新版 HuggingFace 整合
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangGraph 核心
from langgraph.graph import StateGraph, END

# ==========================================
# 2. 配置與設定 (Configuration)
# ==========================================
class Config:
    LLM_MODEL = "gpt-oss:20b" 
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "./data/chroma_db"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
# 定義 Prompt
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# ==========================================
# Define the Output Schema (The Contract)
# ==========================================
class AnalysisResult(BaseModel):
    match_score: int = Field(
        description="A score between 0 and 100 indicating how well the candidate fits the JD."
    )
    missing_critical_skills: List[str] = Field(
        description="List of mandatory technical skills found in the JD but missing in the resume."
    )
    missing_bonus_skills: List[str] = Field(
        description="List of nice-to-have skills or 'bonus' qualifications missing from the resume."
    )
    keyword_optimization_suggestions: List[str] = Field(
        description="Actionable advice on renaming skills or adding specific keywords to pass ATS."
    )
    brief_analysis: str = Field(
        description="A concise, 2-sentence summary of the candidate's suitability."
    )
# ==========================================
# 3. 定義圖學狀態 (Graph State)
# ==========================================
class AgentState(TypedDict):
    """
    這個字典定義了我們在 Graph 中傳遞的數據結構。
    """
    job_description: str  # 輸入的職缺描述 (JD)
    context: str          # 從向量庫檢索到的履歷片段
    analysis: str         # LLM 分析後的最終結果

# ==========================================
# 4. 資料攝取層 (不變，僅修正 Import)
# ==========================================
class ResumeIngestor:
    def __init__(self):
        print(f"🔄 Loading Embedding Model: {Config.EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vector_store = None

    def ingest_resume(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        print(f"📄 Loading Resume: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        print(f"🧩 Split resume into {len(texts)} chunks.")

        print("💾 Creating Vector Store...")
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=Config.VECTOR_DB_PATH
        )
        print("✅ Vector Store Ready!")
        return self.vector_store

# ==========================================
# 5. LangGraph 流程構建 (核心變更)
# ==========================================
class ResumeGraphBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = Ollama(model=Config.LLM_MODEL)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

    # --- Node 1: 檢索 (Retrieve) ---
    def retrieve_node(self, state: AgentState):
        """
        接收 JD，去向量資料庫找相關履歷片段
        """
        print("🔍 Node: Retrieving relevant resume parts...")
        question = state["job_description"]
        documents = self.retriever.invoke(question)
        
        # 將 Document 物件合併成一個字串
        context_str = "\n\n".join([doc.page_content for doc in documents])
        
        # 更新狀態：只更新 context
        return {"context": context_str}

    # --- Node 2: 分析 (Analyze) ---
    # def analyze_node(self, state: AgentState):
    #     """
    #     接收 Context 和 JD，生成分析報告
    #     """
    #     print("🤖 Node: Analyzing fit with LLM...")
        
    #     prompt_template = """
    #     You are an expert Technical Recruiter.
        
    #     Job Description:
    #     {job_description}

    #     Candidate's Experience (from Resume):
    #     {context}

    #     ---
    #     Please provide a structured analysis:
    #     1. **Match Score**: 0-100.
    #     2. **Key Strengths**: 3 matching skills.
    #     3. **Gap Analysis**: Missing keywords.
    #     4. **Verdict**: One sentence summary.
        
    #     Return ONLY the analysis.
    #     """
    #     GAP_ANALYSIS_TEMPLATE = """
    #     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    #     You are an expert Technical Recruiter and Engineering Manager. 
    #     Your goal is to perform a strict "Gap Analysis" between a Job Description (JD) and a Candidate's Resume.
    #     Your analysis must be objective, critical, and based ONLY on the provided text.
    #     <|eot_id|>

    #     <|start_header_id|>user<|end_header_id|>
    #     ### INSTRUCTIONS:
    #     1. **Analyze the JD**: Extract key technical skills (Hard Skills), classifying them into "Must-Have" and "Nice-to-Have".
    #     2. **Analyze the Resume**: Identify technical skills possessed by the candidate based on the context provided.
    #     3. **Compare**: Cross-reference the JD skills against the Resume skills.
    #     - Handle synonyms intelligently (e.g., "K8s" matches "Kubernetes", "AWS" matches "Amazon Web Services").
    #     - If a skill is in the JD but NOT in the Resume, mark it as "MISSING".
    #     - If a skill is in the JD and PARTIALLY covered (e.g., JD asks for "Expert", Resume shows "Junior"), mark as "WEAK".
    #     4. **Output Format**: Return the result purely in JSON format. Do not output any conversational text.

    #     ### CONTEXT DATA:

    #     <JOB_DESCRIPTION>
    #     {job_description}
    #     </JOB_DESCRIPTION>

    #     <RESUME_CONTEXT>
    #     {context}
    #     </RESUME_CONTEXT>

    #     ### REQUIRED JSON OUTPUT FORMAT:
    #     {{
    #         "match_score": <integer_0_to_100>,
    #         "missing_critical_skills": ["skill1", "skill2"],
    #         "missing_bonus_skills": ["skill3"],
    #         "keyword_optimization_suggestions": [
    #             "Suggestion 1: Rename 'X' to 'Y' to match JD",
    #             "Suggestion 2: Explicitly mention 'Z' in the summary"
    #         ],
    #         "brief_analysis": "<2_sentence_summary>"
    #     }}

    #     ### YOUR RESPONSE (JSON ONLY):
    #     <|eot_id|>
    #     <|start_header_id|>assistant<|end_header_id|>
    #     """
        
    #     prompt = PromptTemplate(
    #         template=GAP_ANALYSIS_TEMPLATE,
    #         input_variables=["job_description", "context"]
    #     )

    #     # 建立簡單的 chain: Prompt -> LLM -> StringParser
    #     chain = prompt | self.llm | StrOutputParser()
        
    #     # 執行 Chain
    #     response = chain.invoke({
    #         "job_description": state["job_description"], 
    #         "context": state["context"]
    #     })
        
    #     # 更新狀態：填入 analysis
    #     return {"analysis": response}

    # --- Node 2: Analyze (Refactored for Pydantic V2) ---
    def analyze_node(self, state: AgentState):
        """
        Receives Context and JD, generates a structured Pydantic object, 
        and returns it as a JSON string using Pydantic V2 syntax.
        """
        print("🤖 Node: Analyzing fit with LLM (Structured Output)...")
        
        # 1. Setup the Parser with our Schema
        parser = PydanticOutputParser(pydantic_object=AnalysisResult)

        # 2. Define the Prompt
        GAP_ANALYSIS_TEMPLATE = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert Technical Recruiter. 
        Your goal is to perform a strict "Gap Analysis" between a Job Description (JD) and a Candidate's Resume.
        
        {format_instructions}

        Analyze the following data objectively:
        <|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        ### JOB DESCRIPTION
        {job_description}

        ### RESUME CONTEXT
        {context}

        ### INSTRUCTIONS:
        1. Compare JD skills vs Resume skills.
        2. Handle synonyms intelligently (e.g., K8s = Kubernetes).
        3. Be critical. If a skill is vague, count it as WEAK or MISSING.
        
        ### YOUR RESPONSE (JSON ONLY):
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

        prompt = PromptTemplate(
            template=GAP_ANALYSIS_TEMPLATE,
            input_variables=["job_description", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # 3. Build the Chain
        chain = prompt | self.llm | parser
        
        try:
            # 4. Invoke
            structured_result = chain.invoke({
                "job_description": state["job_description"], 
                "context": state["context"]
            })
            
            # 5. Return (FIXED FOR PYDANTIC V2)
            # We use model_dump_json() instead of json()
            return {"analysis": structured_result.model_dump_json(indent=2)}
            
        except Exception as e:
            print(f"❌ Error parsing output: {e}")
            return {"analysis": f"Error: {str(e)}"}
            
    def build(self):
        """建立並編譯 Graph"""
        workflow = StateGraph(AgentState)

        # 1. 加入節點
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("analyze", self.analyze_node)

        # 2. 定義流程 (Edges)
        workflow.set_entry_point("retrieve")     # 起點 -> Retrieve
        workflow.add_edge("retrieve", "analyze") # Retrieve -> Analyze
        workflow.add_edge("analyze", END)        # Analyze -> 結束

        # 3. 編譯
        return workflow.compile()

# ==========================================
# 6. 主程式
# ==========================================
if __name__ == "__main__":
    dummy_pdf_path = "./data/raw/cv_ver2.pdf"

    if os.path.exists(dummy_pdf_path):
        # 1. 準備資料
        ingestor = ResumeIngestor()
        vector_store = ingestor.ingest_resume(dummy_pdf_path)

        # 2. 準備 Graph
        graph_builder = ResumeGraphBuilder(vector_store)
        app = graph_builder.build()

        # 3. 準備輸入資料
        target_jd = """
        Looking for a Senior Python Developer with:
        - Strong LangGraph and LangChain experience.
        - Knowledge of RAG systems.
        - Experience with Docker and Kubernetes.
        """

        print("\n🚀 Starting LangGraph Workflow...")
        inputs = {"job_description": target_jd}
        
        # 4. 執行 Graph
        # result 會包含最終狀態的所有 key (job_description, context, analysis)
        result = app.invoke(inputs)

        print("\n" + "="*50)
        print("📊 Final Analysis Result")
        print("="*50)
        print(result["analysis"])
        
    else:
        print(f"❌ Please place a resume PDF at '{dummy_pdf_path}' to run.")