import os
import json
import re
import csv
from typing import List, Literal, TypedDict, Dict, Any, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader

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
    missing_critical_skills: List[str] = Field(
        description="List of mandatory technical skills found in the JD but strictly missing in the resume."
    )
    missing_soft_skills: List[str] = Field(
        description="List of soft skills (e.g., leadership, communication) missing from the resume."
    )
    brief_analysis: str = Field(
        description="A concise, 2-sentence summary of the candidate's suitability."
    )

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
    def __init__(self, provider: str, model_name: str, vector_store=None, full_resume_text: str = None, **llm_kwargs):
        self.vector_store = vector_store
        self.full_resume_text = full_resume_text
        
        self.json_llm = LLMFactory.get_llm(provider, model_name, json_mode=True, **llm_kwargs)
        self.text_llm = LLMFactory.get_llm(provider, model_name, json_mode=False, **llm_kwargs)
        
        # Only initialize retriever if a vector store is provided
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
        else:
            self.retriever = None

    def retrieve_node(self, state: AgenticState):
        # --- CHANGED: Conditional Retrieval Logic ---
        if self.full_resume_text:
            print("📄 Node: Loading FULL resume text (RAG bypassed)...")
            return {"context": self.full_resume_text}
        else:
            print("🔍 Node: Retrieving context chunks (RAG enabled)...")
            documents = self.retriever.invoke(state["job_description"])
            context_str = "\n\n".join([doc.page_content for doc in documents])
            return {"context": context_str}

    # def analyze_node(self, state: AgenticState):
    #     current_count = state.get("revision_count", 0)
    #     current_usage = state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        
    #     print(f"🤖 Node: Generating Analysis (Draft {current_count + 1})...")
        
    #     feedback_instruction = ""
    #     if state.get("feedback") and "FAIL" in state.get("feedback", ""):
    #         feedback_instruction = f"\n### CRITIQUE FROM PREVIOUS ATTEMPT:\n{state['feedback']}\nFix these hallucinations."

    #     system_prompt = """You are a Principal Staff Engineer acting as a Technical Recruiter.
    #     Your goal is to perform a gap analysis and classify the candidate's fit.

    #     {feedback_instruction}

    #     ### CLASSIFICATION RULES:
    #     1. **Strong Match**: Candidate possesses 70%+ of the "Must-Have" technical skills.
    #     2. **Good Match**: Candidate possesses 40-70% of "Must-Have" skills.
    #     3. **No Match**: Candidate lacks core technologies (<40% match).

    #     ### JSON OUTPUT INSTRUCTIONS:
    #     You must output EXACTLY AND ONLY a valid JSON object. Do not include markdown formatting or conversational text.
    #     Your JSON must contain EXACTLY these four keys:
    #     - "match_classification": strictly "Strong Match", "Good Match", or "No Match"
    #     - "missing_critical_skills": array of strings
    #     - "missing_soft_skills": array of strings
    #     - "brief_analysis": string, concise 2-sentence summary
    #     """

    #     user_prompt = """### TARGET JOB
    #     Title: {job_title}
    #     Department: {job_department}
    #     Requirements: {job_requirements}

    #     ### CANDIDATE RESUME CONTEXT
    #     {context}

    #     JSON Analysis:"""

    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", system_prompt),
    #         ("user", user_prompt)
    #     ])
        
    #     chain = prompt | self.json_llm 
    #     job_info = state.get("job_info", {})
        
    #     try:
    #         raw_response = chain.invoke({
    #             "job_title": job_info.get("job_title", "N/A"),
    #             "job_department": job_info.get("department", "N/A"),
    #             "job_requirements": job_info.get("requirements", ""),
    #             "context": state["context"],
    #             "feedback_instruction": feedback_instruction
    #         })
            
    #         if hasattr(raw_response, 'response_metadata') and 'token_usage' in raw_response.response_metadata:
    #             usage = raw_response.response_metadata['token_usage']
    #             current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
    #             current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
    #             current_usage["total_tokens"] += usage.get("total_tokens", 0)
            
    #         response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
    #         json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
    #         if not json_match:
    #             raise ValueError(f"Could not find JSON object in output: {response_text[:100]}...")
                
    #         clean_json = json_match.group(0)
    #         parsed_data = json.loads(clean_json)
            
    #         required_keys = ["match_classification", "missing_critical_skills", "missing_soft_skills", "brief_analysis"]
    #         for key in required_keys:
    #             if key not in parsed_data:
    #                 parsed_data[key] = "Data Missing from LLM"

    #         return {
    #             "analysis": json.dumps(parsed_data), 
    #             "revision_count": current_count + 1,
    #             "token_usage": current_usage  # Pass updated tokens back to state
    #         }
            
    #     except Exception as e:
    #         print(f"⚠️ Analysis failed, falling back to default. Error: {e}")
    #         fallback = AnalysisResult(
    #             match_classification="No Match",
    #             missing_critical_skills=["Parsing Error"],
    #             missing_soft_skills=["Parsing Error"],
    #             brief_analysis="System failed to generate valid structured JSON analysis."
    #         )
    #         return {
    #             "analysis": fallback.model_dump_json(), 
    #             "revision_count": current_count + 1,
    #             "token_usage": current_usage
    #         }


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
        Your JSON must contain EXACTLY these four keys:
        - "match_classification": strictly "Strong Match", "Good Match", or "No Match"
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
        
        try:
            raw_response = chain.invoke({
                "job_title": job_info.get("job_title", "N/A"),
                "job_department": job_info.get("department", "N/A"),
                "job_requirements": job_info.get("requirements", ""),
                "context": state["context"],
                "feedback_instruction": feedback_instruction
            })
            
            if hasattr(raw_response, 'response_metadata') and 'token_usage' in raw_response.response_metadata:
                usage = raw_response.response_metadata['token_usage']
                current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                current_usage["total_tokens"] += usage.get("total_tokens", 0)
            
            response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if not json_match:
                raise ValueError(f"Could not find JSON object in output: {response_text[:100]}...")
                
            clean_json = json_match.group(0)
            parsed_data = json.loads(clean_json)
            
            required_keys = ["match_classification", "missing_critical_skills", "missing_soft_skills", "brief_analysis"]
            for key in required_keys:
                if key not in parsed_data:
                    parsed_data[key] = "Data Missing from LLM"

            return {
                "analysis": json.dumps(parsed_data), 
                "revision_count": current_count + 1,
                "token_usage": current_usage
            }
            
        except Exception as e:
            print(f"⚠️ Analysis failed, falling back to default. Error: {e}")
            fallback = AnalysisResult(
                match_classification="No Match",
                missing_critical_skills=["Parsing Error"],
                missing_soft_skills=["Parsing Error"],
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
        
        if "Parsing Error" in state.get("analysis", ""):
            print("   -> Verdict: Skipping audit due to previous parsing error.")
            return {"feedback": "PASS"}
            
        reflection_prompt = ChatPromptTemplate.from_template(
            "You are an auditing algorithm. Review the generated JSON analysis against the resume context. "
            "If the analysis claims a critical skill or soft skill is MISSING, but it ACTUALLY EXISTS in the context, output: 'FAIL: You claimed [Skill] is missing, but it is in the context.' "
            "If the analysis claims a skill is MATCHED, but it is NOT in the context, output: 'FAIL: You hallucinated [Skill].' "
            "If the analysis is accurate based strictly on the text, output exactly: 'PASS'."
            "\n\nContext:\n{context}\n\nAnalysis:\n{analysis}"
        )
        
        chain = reflection_prompt | self.text_llm 
        feedback_msg = chain.invoke({"context": state["context"], "analysis": state["analysis"]})
        
        # --- CHANGED: Accumulate Token Usage for the critique step ---
        if hasattr(feedback_msg, 'response_metadata') and 'token_usage' in feedback_msg.response_metadata:
            usage = feedback_msg.response_metadata['token_usage']
            current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            current_usage["total_tokens"] += usage.get("total_tokens", 0)
        
        feedback_text = feedback_msg.content if hasattr(feedback_msg, 'content') else str(feedback_msg)
        print(f"   -> Verdict: {feedback_text.strip()}")
        
        return {
            "feedback": feedback_text.strip(),
            "token_usage": current_usage # Pass updated tokens back to state
        }

    def route_to_revision(self, state: AgenticState) -> str:
        if state.get("revision_count", 0) >= 3:
            return END
        if "FAIL" in state.get("feedback", ""):
            print("🔄 Hallucination detected! Routing back to Analyze Node...")
            return "analyze"
        return END

    def build(self):
        workflow = StateGraph(AgenticState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("critique", self.critique_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", "critique")
        workflow.add_conditional_edges("critique", self.route_to_revision, {"analyze": "analyze", END: END})
        
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
        ENGINE = getattr(Config, 'GPT_MODEL', 'gpt-4o')
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
            vector_store=vector_store  # Pass vector store
        ).build()

    print(f"\n📂 Loading Job Descriptions from {Config.JOBS_FILE}...")
    with open(Config.JOBS_FILE, 'r', encoding='utf-8') as f:
        job_list = json.load(f)

    results_summary = []

    print(f"🚀 Starting Multi-Provider Pipeline ({PROVIDER.upper()} / {ENGINE}) for {len(job_list)} jobs...\n")

    for i, job in enumerate(job_list):
        jd_query = f"{job.get('job_title', '')} {job.get('requirements', '')}"
        inputs = {
            "job_info": job,
            "job_description": jd_query,
            "context": "",
            "revision_count": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} 
        }

        print(f"\n--- Processing Job {i+1}/{len(job_list)}: {job.get('job_title')} ---")
        output = app.invoke(inputs)
        
        try:
            result_json = json.loads(output["analysis"])
        except Exception as e:
            result_json = {
                "match_classification": "No Match",
                "missing_critical_skills": ["Parsing Error"],
                "missing_soft_skills": ["Parsing Error"],
                "brief_analysis": f"Error loading JSON: {str(e)}"
            }
        
        classification = result_json.get('match_classification', 'No Match')
        print(f"[{i+1}/{len(job_list)}] Result: {classification.upper()}")
        print(f"Brief Analysis: {result_json.get('brief_analysis', '')}")
        
        # --- CHANGED: Log token usage safely to the console if it exists ---
        final_tokens = output.get("token_usage", {})
        if final_tokens.get("total_tokens", 0) > 0:
            print(f"☁️  API Tokens Used -> Prompt: {final_tokens.get('prompt_tokens')} | Completion: {final_tokens.get('completion_tokens')} | Total: {final_tokens.get('total_tokens')}")
        
        results_summary.append({
            "job_title": job.get('job_title', 'Unknown'),
            "classification": classification,
            "missing_critical_skills": result_json.get('missing_critical_skills', []),
            "missing_soft_skills": result_json.get('missing_soft_skills', []),
            "brief_analysis": result_json.get('brief_analysis', ''),
            "total_tokens_used": final_tokens.get("total_tokens", 0) # Track in export
        })

    # ==========================================
    # CSV EXPORT LOGIC
    # ==========================================
    csv_filename = "multi_provider_" + getattr(Config, 'ANALYSIS_OUTPUT_CSV', 'analysis_results.csv')
    output_dir = getattr(Config, 'OUTPUT_DIR', './data/output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_path = os.path.join(output_dir, csv_filename)

    print(f"\n💾 Saving all {len(results_summary)} results to CSV: {csv_path}")
    
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = [
                'Job Title', 
                'Classification', 
                'Missing Critical Skills', 
                'Missing Soft Skills',
                'Brief Analysis',
                'Total Tokens Used' # Added to CSV tracking
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in results_summary:
                def flatten_list(lst):
                    if not lst: return ""
                    return ", ".join(lst) if isinstance(lst, list) else str(lst)

                writer.writerow({
                    'Job Title': res['job_title'],
                    'Classification': res['classification'],
                    'Missing Critical Skills': flatten_list(res['missing_critical_skills']),
                    'Missing Soft Skills': flatten_list(res['missing_soft_skills']),
                    'Brief Analysis': res['brief_analysis'],
                    'Total Tokens Used': res['total_tokens_used']
                })
        print("✅ CSV export complete.")
        
    except Exception as e:
        print(f"❌ Error writing CSV: {e}")

    print("\n✅ Done.")