import os
import json
import re
import csv
from typing import List, Literal, TypedDict, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Provider imports for the Factory
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Import your existing utilities
from .utils import Config, ResumeIngestor

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

# ==========================================
# 3. LLM Provider Factory (UPDATED: Added json_mode flag)
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
            # Only apply format="json" if requested by the node
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
    def __init__(self, vector_store, provider: str, model_name: str, **llm_kwargs):
        self.vector_store = vector_store
        
        # Instantiate two separate models based on the required task output
        self.json_llm = LLMFactory.get_llm(provider, model_name, json_mode=False, **llm_kwargs)
        self.text_llm = LLMFactory.get_llm(provider, model_name, json_mode=False, **llm_kwargs)
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})

    def retrieve_node(self, state: AgenticState):
        print("🔍 Node: Retrieving context...")
        documents = self.retriever.invoke(state["job_description"])
        context_str = "\n\n".join([doc.page_content for doc in documents])
        return {"context": context_str}

    def analyze_node(self, state: AgenticState):
        current_count = state.get("revision_count", 0)
        print(f"🤖 Node: Generating Analysis (Draft {current_count + 1})...")
        
        feedback_instruction = ""
        if state.get("feedback") and "FAIL" in state.get("feedback", ""):
            feedback_instruction = f"\n### CRITIQUE FROM PREVIOUS ATTEMPT:\n{state['feedback']}\nFix these hallucinations."

        # FIXED SYSTEM PROMPT: Manually defining the schema so the local model understands it easily
        system_prompt = """You are a Principal Staff Engineer acting as a Technical Recruiter.
        Your goal is to perform a gap analysis and classify the candidate's fit.

        {feedback_instruction}

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

        user_prompt = """### TARGET JOB
        Title: {job_title}
        Department: {job_department}
        Requirements: {job_requirements}

        ### CANDIDATE RESUME CONTEXT
        {context}

        JSON Analysis:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        
        # REMOVED the Pydantic parser from the chain
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
            
            # Get the raw text output
            response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            # THE FIX: Bulletproof Regex JSON Extraction
            # This hunts for anything between { and } even if the model chatted first
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if not json_match:
                raise ValueError(f"Could not find JSON object in output: {response_text[:100]}...")
                
            clean_json = json_match.group(0)
            parsed_data = json.loads(clean_json)
            
            # Ensure required keys exist just in case the model missed one
            required_keys = ["match_classification", "missing_critical_skills", "missing_soft_skills", "brief_analysis"]
            for key in required_keys:
                if key not in parsed_data:
                    parsed_data[key] = "Data Missing from LLM"

            # Serialize back to JSON string for the graph state
            return {"analysis": json.dumps(parsed_data), "revision_count": current_count + 1}
            
        except Exception as e:
            print(f"⚠️ Analysis failed, falling back to default. Error: {e}")
            fallback = AnalysisResult(
                match_classification="No Match",
                missing_critical_skills=["Parsing Error"],
                missing_soft_skills=["Parsing Error"],
                brief_analysis="System failed to generate valid structured JSON analysis."
            )
            return {"analysis": fallback.model_dump_json(), "revision_count": current_count + 1}
    def critique_node(self, state: AgenticState):
        print("🧐 Node: Auditing Output for Hallucinations...")
        
        # CRITICAL FIX: If the analysis failed in the previous step, don't audit it.
        # Just pass the failure forward to avoid an infinite hallucination loop.
        if "Parsing Error" in state.get("analysis", ""):
            print("   -> Verdict: Skipping audit due to previous parsing error.")
            return {"feedback": "PASS"} # Let it exit the graph gracefully
            
        reflection_prompt = ChatPromptTemplate.from_template(
            "You are an auditing algorithm. Review the generated JSON analysis against the resume context. "
            "If the analysis claims a critical skill or soft skill is MISSING, but it ACTUALLY EXISTS in the context, output: 'FAIL: You claimed [Skill] is missing, but it is in the context.' "
            "If the analysis claims a skill is MATCHED, but it is NOT in the context, output: 'FAIL: You hallucinated [Skill].' "
            "If the analysis is accurate based strictly on the text, output exactly: 'PASS'."
            "\n\nContext:\n{context}\n\nAnalysis:\n{analysis}"
        )
        
        chain = reflection_prompt | self.text_llm 
        feedback_msg = chain.invoke({"context": state["context"], "analysis": state["analysis"]})
        
        feedback_text = feedback_msg.content if hasattr(feedback_msg, 'content') else str(feedback_msg)
        print(f"   -> Verdict: {feedback_text.strip()}")
        return {"feedback": feedback_text.strip()}

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
    # 1. READ PROVIDER AND ENGINE DIRECTLY FROM JSON CONFIG
    PROVIDER = getattr(Config, 'LLM_PROVIDER', 'ollama')
    ENGINE = getattr(Config, 'LLM_MODEL', 'gpt-oss:20b')
    
    if not os.path.exists(Config.RESUME_FILE):
        print(f"❌ Error: Resume file '{Config.RESUME_FILE}' not found.")
        exit(1)

    ingestor = ResumeIngestor()
    vector_store = ingestor.get_vector_store(Config.RESUME_FILE)
    
    app = MultiProviderResumeGraphBuilder(
        vector_store=vector_store, 
        provider=PROVIDER, 
        model_name=ENGINE
    ).build()

    with open(Config.JOBS_FILE, 'r', encoding='utf-8') as f:
        job_list = json.load(f)

    # List to store results for CSV export
    results_summary = []

    print(f"🚀 Starting Multi-Provider Pipeline ({PROVIDER} / {ENGINE}) for {len(job_list)} jobs...\n")

    for i, job in enumerate(job_list):
        jd_query = f"{job.get('job_title', '')} {job.get('requirements', '')}"
        inputs = {
            "job_info": job,
            "job_description": jd_query,
            "context": "",
            "revision_count": 0
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
        
        # Add result to our summary list
        results_summary.append({
            "job_title": job.get('job_title', 'Unknown'),
            "classification": classification,
            "missing_critical_skills": result_json.get('missing_critical_skills', []),
            "missing_soft_skills": result_json.get('missing_soft_skills', []),
            "brief_analysis": result_json.get('brief_analysis', '')
        })

    # ==========================================
    # CSV EXPORT LOGIC
    # ==========================================
    csv_filename = "multi_provider_" + getattr(Config, 'ANALYSIS_OUTPUT_CSV', 'analysis_results.csv')
    output_dir = getattr(Config, 'OUTPUT_DIR', './data/output')
    
    # Ensure the output directory exists
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
                'Brief Analysis'
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in results_summary:
                # Helper to flatten lists safely 
                def flatten_list(lst):
                    if not lst: return ""
                    return ", ".join(lst) if isinstance(lst, list) else str(lst)

                writer.writerow({
                    'Job Title': res['job_title'],
                    'Classification': res['classification'],
                    'Missing Critical Skills': flatten_list(res['missing_critical_skills']),
                    'Missing Soft Skills': flatten_list(res['missing_soft_skills']),
                    'Brief Analysis': res['brief_analysis']
                })
        print("✅ CSV export complete.")
        
    except Exception as e:
        print(f"❌ Error writing CSV: {e}")

    print("\n✅ Done.")