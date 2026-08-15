import os
import json
import re
import csv
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal, TypedDict, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

from dotenv import load_dotenv

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever

# Provider imports for the Factory
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Import your existing utilities
from .utils import Config, ResumeIngestor

load_dotenv()

# ==========================================
# 1. Pydantic Output Schema (Per Specification)
# ==========================================
class AnalysisResult(BaseModel):
    match_classification: Literal["Strong Match", "Good Match", "No Match"] = Field(
        description="The categorical fit based on the rules provided."
    )
    matched_critical_skills: List[str] = Field(
        default_factory=list,
        description="List of mandatory technical skills from the JD that ARE present in the resume."
    )
    missing_critical_skills: List[str] = Field(
        description="List of mandatory technical skills found in the JD but strictly missing in the resume."
    )
    missing_soft_skills: List[str] = Field(
        description="List of soft skills (e.g., leadership, communication) missing from the resume."
    )
    brief_analysis: str = Field(
        description="A concise, 2-sentence summary of the candidate's suitability."
    )


class CritiqueResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reason: str = ""


def verify_classification(parsed_data: dict) -> dict:
    """Override the LLM's stated classification when it contradicts its own skill counts."""
    matched = len(parsed_data.get("matched_critical_skills") or [])
    missing = len(parsed_data.get("missing_critical_skills") or [])
    total = matched + missing
    if total == 0:
        return parsed_data

    ratio = matched / total
    expected = "Strong Match" if ratio >= 0.7 else "Good Match" if ratio >= 0.4 else "No Match"
    if parsed_data.get("match_classification") != expected:
        print(f"⚠️ Classification mismatch: LLM said {parsed_data.get('match_classification')}, "
              f"counts imply {expected} ({matched}/{total}). Overriding.")
        parsed_data["match_classification"] = expected
    return parsed_data


def decompose_requirements(requirements: str) -> List[str]:
    """Split a job's requirements into separate retrieval queries.

    One blended embedding of the whole JD skews toward whichever skill dominates the text,
    starving the rest; querying each skill separately gives every requirement its own budget.
    """
    if not requirements:
        return []
    parts = re.split(r"[,;\n]|\band\b|、", requirements)
    return [p.strip() for p in parts if p.strip()]


def filter_by_score(scored_docs, min_score: float):
    return [doc for doc, score in scored_docs if score >= min_score]


def job_key(job: Dict[str, Any]) -> tuple:
    """Titles alone collide across postings, so resume-from-checkpoint keys on title + department."""
    return (job.get("job_title", "Unknown"), job.get("department", ""))


def load_completed_jobs(csv_path: str) -> set:
    """Read an existing results CSV so an interrupted batch can resume where it stopped."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline='', encoding='utf-8') as f:
        return {(row.get("Job Title", ""), row.get("Department", "")) for row in csv.DictReader(f)}


def _build_retryable_exceptions():
    retryable = (ConnectionError, TimeoutError)
    try:
        import openai
        retryable += (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
    except (ImportError, AttributeError):
        pass
    return retryable


RETRYABLE_EXCEPTIONS = _build_retryable_exceptions()

# ==========================================
# 2. Graph State
# ==========================================
class AgenticState(TypedDict):
    job_description: str
    context: str
    analysis: str
    feedback: str
    revision_count: int
    job_info: Dict[str, Any]
    # --- CHANGED: Added token_usage dictionary to accumulate cloud tokens ---
    token_usage: Dict[str, int]

# ==========================================
# 3. LLM Provider Factory 
# ==========================================
class LLMFactory:
    @staticmethod
    def get_llm(provider: str, model_name: str, json_mode: bool = False, **kwargs):
        provider = provider.lower()
        temperature = kwargs.get("temperature", 0.1)

        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
            
            model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            
            return ChatOpenAI(
                model=model_name, 
                temperature=temperature,
                model_kwargs=model_kwargs
            )
            
        elif provider == "ollama":
            base_url = kwargs.get("base_url", "http://localhost:11434")
            return ChatOllama(
                model=model_name, 
                temperature=temperature, 
                base_url=base_url, 
                **({"format": "json"} if json_mode else {})
            )
            
        elif provider == "vllm":
            base_url = kwargs.get("base_url", "http://localhost:8000/v1")
            api_key = os.getenv("VLLM_API_KEY", "EMPTY") 
            model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            return ChatOpenAI(
                model=model_name, 
                temperature=temperature, 
                openai_api_base=base_url, 
                openai_api_key=api_key,
                model_kwargs=model_kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

# ==========================================
# 4. Agentic LangGraph Workflow
# ==========================================
class MultiProviderResumeGraphBuilder:
    # --- CHANGED: Accept either vector_store (for RAG) or full_resume_text (for No-RAG) ---
    def __init__(self, provider: str, model_name: str, vector_store=None, full_resume_text: str = None,
                 resume_texts=None, **llm_kwargs):
        self.vector_store = vector_store
        self.full_resume_text = full_resume_text

        self.json_llm = LLMFactory.get_llm(provider, model_name, json_mode=True, **llm_kwargs)
        self.text_llm = LLMFactory.get_llm(provider, model_name, json_mode=False, **llm_kwargs)

        # Chroma's SQLite backend is not safe for concurrent reads from the batch pool.
        self.retrieval_lock = threading.Lock()
        self.bm25_retriever = None
        self._base_k = Config.RETRIEVER_K

        if self.vector_store:
            try:
                total_chunks = self.vector_store._collection.count()
            except Exception:
                total_chunks = 0
            if total_chunks:
                self._base_k = min(total_chunks, max(Config.RETRIEVER_K, total_chunks // 3))
            if resume_texts:
                self.bm25_retriever = BM25Retriever.from_documents(resume_texts)
                self.bm25_retriever.k = self._base_k

    def retrieve_node(self, state: AgenticState):
        # --- CHANGED: Conditional Retrieval Logic ---
        if self.full_resume_text:
            print("📄 Node: Loading FULL resume text (RAG bypassed)...")
            return {"context": self.full_resume_text}

        print("🔍 Node: Retrieving context chunks (RAG enabled)...")
        job_info = state.get("job_info", {})
        skill_queries = decompose_requirements(job_info.get("requirements", ""))
        if not skill_queries:
            skill_queries = [state["job_description"]]

        feedback = state.get("feedback", "")
        if feedback.startswith("FAIL"):
            skill_queries.append(feedback)

        # Re-entry after a failed critique must widen the context, not replace it —
        # discarding the first pass's chunks can make revision 2 worse than revision 1.
        prior_context = re.sub(r"\n*\[UNRESOLVED —[^\]]*\]", "", state.get("context", "")).strip()
        seen_content = set(prior_context.split("\n\n")) if prior_context else set()
        collected_chunks = []
        unresolved_skills = []

        for query in skill_queries:
            with self.retrieval_lock:
                scored = self.vector_store.similarity_search_with_relevance_scores(query, k=self._base_k)
                hits = filter_by_score(scored, Config.RETRIEVER_MIN_SCORE)
                if self.bm25_retriever:
                    hits = hits + self.bm25_retriever.invoke(query)

            if not hits:
                unresolved_skills.append(query)
                continue

            for doc in hits:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    collected_chunks.append(doc.page_content)

        parts = ([prior_context] if prior_context else []) + collected_chunks
        context_str = "\n\n".join(parts)
        if unresolved_skills:
            context_str += (f"\n\n[UNRESOLVED — no resume evidence above the relevance threshold for: "
                            f"{', '.join(unresolved_skills)}]")
        print(f"   -> {len(collected_chunks)} new chunk(s), {len(unresolved_skills)} unresolved requirement(s).")
        return {"context": context_str}

    def analyze_node(self, state: AgenticState):
        current_count = state.get("revision_count", 0)
        current_usage = state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        
        print(f"🤖 Node: Generating Analysis (Draft {current_count + 1})...")
        
        feedback_instruction = ""
        if state.get("feedback") and "FAIL" in state.get("feedback", ""):
            feedback_instruction = f"\n### CRITIQUE FROM PREVIOUS ATTEMPT:\n{state['feedback']}\nFix these hallucinations."

        # =================================================================
        # CACHE-OPTIMIZED PROMPT STRUCTURE
        # 1. Put the massive, static Resume Context at the very top.
        # 2. Put the static rules and instructions immediately after.
        # =================================================================
        system_prompt = """You are a Principal Staff Engineer acting as a Technical Recruiter.
        Your goal is to perform a gap analysis and classify the candidate's fit based strictly on the provided resume context.

        ### CANDIDATE RESUME CONTEXT (STATIC)
        {context}

        ### CLASSIFICATION RULES:
        1. **Strong Match**: Candidate possesses 70%+ of the "Must-Have" technical skills.
        2. **Good Match**: Candidate possesses 40-70% of "Must-Have" skills.
        3. **No Match**: Candidate lacks core technologies (<40% match).

        ### JSON OUTPUT INSTRUCTIONS:
        You must output EXACTLY AND ONLY a valid JSON object. Do not include markdown formatting or conversational text.
        Your JSON must contain EXACTLY these five keys:
        - "match_classification": strictly "Strong Match", "Good Match", or "No Match"
        - "matched_critical_skills": array of strings, the mandatory technical skills from the JD that ARE in the resume
        - "missing_critical_skills": array of strings
        - "missing_soft_skills": array of strings
        - "brief_analysis": string, concise 2-sentence summary
        """

        # =================================================================
        # 3. Put the dynamic variables (Job Info & Feedback) at the very end.
        # =================================================================
        user_prompt = """{feedback_instruction}

        ### TARGET JOB (DYNAMIC)
        Title: {job_title}
        Department: {job_department}
        Requirements: {job_requirements}

        JSON Analysis:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        
        chain = prompt | self.json_llm 
        job_info = state.get("job_info", {})
        
        raw_response = None
        for attempt in range(3):
            try:
                raw_response = chain.invoke({
                    "job_title": job_info.get("job_title", "N/A"),
                    "job_department": job_info.get("department", "N/A"),
                    "job_requirements": job_info.get("requirements", ""),
                    "context": state["context"],
                    "feedback_instruction": feedback_instruction
                })
                break
            except RETRYABLE_EXCEPTIONS as e:
                wait = 2 ** attempt
                print(f"⚠️ Transient provider error ({e.__class__.__name__}), retrying in {wait}s...")
                time.sleep(wait)

        if raw_response is None:
            fallback = AnalysisResult(
                match_classification="No Match",
                matched_critical_skills=[],
                missing_critical_skills=["API_UNAVAILABLE"],
                missing_soft_skills=["API_UNAVAILABLE"],
                brief_analysis="LLM provider unreachable after retries; this is an infrastructure failure, not a candidate assessment."
            )
            return {
                "analysis": fallback.model_dump_json(),
                "revision_count": current_count + 1,
                "token_usage": current_usage
            }

        if hasattr(raw_response, 'response_metadata') and 'token_usage' in raw_response.response_metadata:
            usage = raw_response.response_metadata['token_usage']
            current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            current_usage["total_tokens"] += usage.get("total_tokens", 0)

        try:
            response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError(f"Could not find JSON object in output: {response_text[:100]}...")

            parsed_data = json.loads(json_match.group(0))

            for key in ["matched_critical_skills", "missing_critical_skills", "missing_soft_skills"]:
                if not isinstance(parsed_data.get(key), list):
                    parsed_data[key] = []
            parsed_data.setdefault("brief_analysis", "Data Missing from LLM")
            parsed_data = verify_classification(parsed_data)

            return {
                "analysis": json.dumps(parsed_data),
                "revision_count": current_count + 1,
                "token_usage": current_usage
            }

        except (ValueError, json.JSONDecodeError) as e:
            print(f"⚠️ Could not parse analysis JSON. Error: {e}")
            fallback = AnalysisResult(
                match_classification="No Match",
                matched_critical_skills=[],
                missing_critical_skills=["PARSE_ERROR"],
                missing_soft_skills=["PARSE_ERROR"],
                brief_analysis="System failed to generate valid structured JSON analysis."
            )
            return {
                "analysis": fallback.model_dump_json(),
                "revision_count": current_count + 1,
                "token_usage": current_usage
            }

    def critique_node(self, state: AgenticState):
        print("🧐 Node: Auditing Output for Hallucinations...")
        current_usage = state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        
        if "API_UNAVAILABLE" in state.get("analysis", ""):
            print("   -> Verdict: Provider unreachable; no audit possible.")
            return {"feedback": "PASS"}

        if "PARSE_ERROR" in state.get("analysis", ""):
            print("   -> Verdict: Unparseable analysis, forcing a retry.")
            return {"feedback": "FAIL: Previous response was not valid JSON. Follow the JSON OUTPUT INSTRUCTIONS exactly."}

        reflection_prompt = ChatPromptTemplate.from_template(
            "You are an auditing algorithm. Review the generated JSON analysis against the resume context. "
            "Respond with a JSON object containing exactly two keys: "
            '"verdict" (strictly "PASS" or "FAIL") and "reason" (a short string). '
            'Set verdict to "FAIL" if the analysis claims a skill is MISSING but it actually EXISTS in the context, '
            'or claims a skill is MATCHED but it is NOT in the context. Explain which skill in "reason". '
            'Otherwise set verdict to "PASS" with an empty reason.'
            "\n\nContext:\n{context}\n\nAnalysis:\n{analysis}"
        )

        chain = reflection_prompt | self.json_llm
        feedback_msg = chain.invoke({"context": state["context"], "analysis": state["analysis"]})

        # --- CHANGED: Accumulate Token Usage for the critique step ---
        if hasattr(feedback_msg, 'response_metadata') and 'token_usage' in feedback_msg.response_metadata:
            usage = feedback_msg.response_metadata['token_usage']
            current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            current_usage["total_tokens"] += usage.get("total_tokens", 0)

        response_text = feedback_msg.content if hasattr(feedback_msg, 'content') else str(feedback_msg)

        # json_mode guarantees valid JSON, not this specific schema.
        try:
            result = CritiqueResult.model_validate_json(response_text)
        except (ValidationError, ValueError):
            result = CritiqueResult(verdict="FAIL", reason="Critique response did not match the expected schema.")

        feedback_text = "PASS" if result.verdict == "PASS" else f"FAIL: {result.reason}"
        print(f"   -> Verdict: {feedback_text}")

        return {
            "feedback": feedback_text,
            "token_usage": current_usage # Pass updated tokens back to state
        }

    def route_to_revision(self, state: AgenticState) -> str:
        if state.get("revision_count", 0) >= 3:
            return END
        if state.get("feedback", "").startswith("FAIL"):
            print("🔄 Hallucination detected! Re-retrieving context with the critique as a hint...")
            return "retrieve"
        return END

    def build(self):
        workflow = StateGraph(AgenticState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("critique", self.critique_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", "critique")
        workflow.add_conditional_edges("critique", self.route_to_revision, {"retrieve": "retrieve", END: END})
        
        return workflow.compile()

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    
    PROVIDER = getattr(Config, 'LLM_PROVIDER', 'ollama').lower()
    
    if not os.path.exists(Config.RESUME_FILE):
        print(f"❌ Error: Resume file '{Config.RESUME_FILE}' not found.")
        exit(1)

    # --- CHANGED: Dynamic App Initialization (RAG vs No-RAG) ---
    if PROVIDER == "openai":
        ENGINE = getattr(Config, 'GPT_MODEL', 'gpt-5-nano')
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Configuration Error: 'LLM_PROVIDER' is set to 'openai' in config.json, but OPENAI_API_KEY is missing in your .env file.")
            exit(1)
            
        print("☁️ Cloud Provider detected: Disabling RAG and loading full resume text...")
        loader = PyPDFLoader(Config.RESUME_FILE)
        docs = loader.load()
        full_text = "\n\n".join([doc.page_content for doc in docs])
        
        app = MultiProviderResumeGraphBuilder(
            provider=PROVIDER, 
            model_name=ENGINE,
            full_resume_text=full_text  # Pass raw text, no vector store
        ).build()
        
    else:
        ENGINE = getattr(Config, 'LLM_MODEL', 'gpt-oss:20b')
        print("💻 Local Provider detected: Enabling RAG (Vector Search)...")
        ingestor = ResumeIngestor()
        vector_store = ingestor.get_vector_store(Config.RESUME_FILE)
        
        app = MultiProviderResumeGraphBuilder(
            provider=PROVIDER,
            model_name=ENGINE,
            vector_store=vector_store,
            resume_texts=ingestor.texts
        ).build()

    print(f"\n📂 Loading Job Descriptions from {Config.JOBS_FILE}...")
    with open(Config.JOBS_FILE, 'r', encoding='utf-8') as f:
        job_list = json.load(f)

    output_dir = getattr(Config, 'OUTPUT_DIR', './data/output')
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = "multi_provider_" + getattr(Config, 'ANALYSIS_OUTPUT_CSV', 'analysis_results.csv')
    csv_path = os.path.join(output_dir, csv_filename)

    fieldnames = [
        'Job Title',
        'Department',
        'Classification',
        'Matched Critical Skills',
        'Missing Critical Skills',
        'Missing Soft Skills',
        'Brief Analysis',
        'Total Tokens Used'
    ]

    already_done = load_completed_jobs(csv_path)
    pending = [job for job in job_list if job_key(job) not in already_done]
    if already_done:
        print(f"\u23ed\ufe0f  Resuming: {len(already_done)} job(s) already in {csv_path}, {len(pending)} remaining.")

    token_report = []
    csv_lock = threading.Lock()
    counter_lock = threading.Lock()
    completed = 0

    def flatten_list(lst):
        if not lst:
            return ""
        return ", ".join(lst) if isinstance(lst, list) else str(lst)

    def process_job(job):
        jd_query = f"{job.get('job_title', '')} {job.get('requirements', '')}"
        inputs = {
            "job_info": job,
            "job_description": jd_query,
            "context": "",
            "revision_count": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        output = app.invoke(inputs)

        try:
            result_json = json.loads(output["analysis"])
        except (ValueError, json.JSONDecodeError) as e:
            result_json = {
                "match_classification": "No Match",
                "matched_critical_skills": [],
                "missing_critical_skills": ["PARSE_ERROR"],
                "missing_soft_skills": ["PARSE_ERROR"],
                "brief_analysis": f"Error loading JSON: {e}"
            }
        return job, result_json, output.get("token_usage", {})

    print(f"\U0001f680 Starting Multi-Provider Pipeline ({PROVIDER.upper()} / {ENGINE}) "
          f"for {len(pending)} job(s) with {Config.MAX_WORKERS} worker(s)...\n")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            csv_file.flush()

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as pool:
            futures = [pool.submit(process_job, job) for job in pending]
            for future in as_completed(futures):
                try:
                    job, result_json, tokens = future.result()
                except Exception as e:
                    print(f"\u274c Job failed unexpectedly: {e}")
                    continue

                with counter_lock:
                    completed += 1
                    position = completed

                classification = result_json.get('match_classification', 'No Match')
                print(f"[{position}/{len(pending)}] {classification.upper()} : {job.get('job_title')}")

                token_report.append({
                    "Job Title": job.get('job_title', 'Unknown'),
                    "Prompt Tokens": tokens.get("prompt_tokens", 0),
                    "Completion Tokens": tokens.get("completion_tokens", 0),
                    "Total Tokens": tokens.get("total_tokens", 0)
                })

                # Flush per row so a crash mid-batch keeps every finished job.
                with csv_lock:
                    writer.writerow({
                        'Job Title': job.get('job_title', 'Unknown'),
                        'Department': job.get('department', ''),
                        'Classification': classification,
                        'Matched Critical Skills': flatten_list(result_json.get('matched_critical_skills', [])),
                        'Missing Critical Skills': flatten_list(result_json.get('missing_critical_skills', [])),
                        'Missing Soft Skills': flatten_list(result_json.get('missing_soft_skills', [])),
                        'Brief Analysis': result_json.get('brief_analysis', ''),
                        'Total Tokens Used': tokens.get("total_tokens", 0)
                    })
                    csv_file.flush()

    print(f"\n\u2705 Results written to {csv_path}")

    if PROVIDER == "openai" and token_report:
        token_csv_path = os.path.join(output_dir, "token_usage_report.csv")
        print(f"\U0001f4be Saving token usage report to: {token_csv_path}")
        try:
            with open(token_csv_path, mode='w', newline='', encoding='utf-8') as token_file:
                token_writer = csv.DictWriter(
                    token_file, fieldnames=['Job Title', 'Prompt Tokens', 'Completion Tokens', 'Total Tokens']
                )
                token_writer.writeheader()
                token_writer.writerows(token_report)
            print("\u2705 Token report export complete.")
        except OSError as e:
            print(f"\u274c Error writing Token CSV: {e}")

    print("\n\u2705 Done.")
