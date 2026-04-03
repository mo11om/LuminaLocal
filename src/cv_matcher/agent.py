import os
import json
import csv
from typing import List, Literal, TypedDict, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
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
# 3. LLM Provider Factory (CRITICAL REQUIREMENT)
# ==========================================
class LLMFactory:
    @staticmethod
    def get_llm(provider: str, model_name: str, **kwargs):
        provider = provider.lower()
        temperature = kwargs.get("temperature", 0.1)

        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
            # OpenAI can also be forced into JSON mode by passing model_kwargs
            return ChatOpenAI(
                model=model_name, 
                temperature=temperature,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            
        elif provider == "ollama":
            base_url = kwargs.get("base_url", "http://localhost:11434")
            # CRITICAL FIX: Add format="json" here
            return ChatOllama(
                model=model_name, 
                temperature=temperature, 
                base_url=base_url, 
                format="json" 
            )
            
        elif provider == "vllm":
            base_url = kwargs.get("base_url", "http://localhost:8000/v1")
            api_key = os.getenv("VLLM_API_KEY", "EMPTY") 
            return ChatOpenAI(
                model=model_name, 
                temperature=temperature, 
                openai_api_base=base_url, 
                openai_api_key=api_key,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
# ==========================================
# 4. Agentic LangGraph Workflow
# ==========================================
class MultiProviderResumeGraphBuilder:
    def __init__(self, vector_store, provider: str, model_name: str, **llm_kwargs):
        self.vector_store = vector_store
        self.llm = LLMFactory.get_llm(provider, model_name, **llm_kwargs)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})

    def retrieve_node(self, state: AgenticState):
        print("🔍 Node: Retrieving context...")
        documents = self.retriever.invoke(state["job_description"])
        context_str = "\n\n".join([doc.page_content for doc in documents])
        return {"context": context_str}

    def analyze_node(self, state: AgenticState):
        current_count = state.get("revision_count", 0)
        print(f"🤖 Node: Generating Analysis (Draft {current_count + 1})...")
        
        parser = PydanticOutputParser(pydantic_object=AnalysisResult)
        
        # Inject feedback from the critique node if this is a revision
        feedback_instruction = ""
        if state.get("feedback") and "FAIL" in state.get("feedback", ""):
            feedback_instruction = f"\n### CRITIQUE FROM PREVIOUS ATTEMPT:\n{state['feedback']}\nFix these hallucinations in your new JSON output."

        
        PROMPT_TEMPLATE = """You are a Principal Staff Engineer acting as a Technical Recruiter.
        Your goal is to perform a gap analysis and classify the candidate's fit.

        {format_instructions}
        {feedback_instruction}

        ### CLASSIFICATION RULES:
        1. **Strong Match**: Candidate possesses 70%+ of the "Must-Have" technical skills found in the JD.
        2. **Good Match**: Candidate possesses 40-70% of "Must-Have" skills or has strong transferrable skills.
        3. **No Match**: Candidate lacks significant core technologies required (e.g., <40% match) or has a completely irrelevant background.

        ### INSTRUCTIONS:
        - Handle synonyms intelligently (e.g., AWS vs Amazon Web Services).
        - DO NOT hallucinate skills that are not explicitly found in the RESUME CONTEXT.
        - CRITICAL: Output ONLY a valid JSON data instance containing your final analysis. DO NOT output the JSON schema definition. DO NOT include keys like "properties" or "type".

        ### EXAMPLE OF REQUIRED JSON INSTANCE OUTPUT:
        {{
            "match_classification": "Good Match",
            "missing_critical_skills": ["Kubernetes", "Docker"],
            "missing_soft_skills": ["Agile Leadership"],
            "brief_analysis": "The candidate has strong Python skills but lacks required Kubernetes experience."
        }}

        ### TARGET JOB
        Title: {job_title}
        Department: {job_department}
        Requirements: {job_requirements}

        ### CANDIDATE RESUME CONTEXT
        {context}

        ### YOUR JSON RESPONSE:
        """

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["job_title", "job_department", "job_requirements", "context", "feedback_instruction"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.llm | parser
        job_info = state.get("job_info", {})

        try:
            structured_result = chain.invoke({
                "job_title": job_info.get("job_title", "N/A"),
                "job_department": job_info.get("department", "N/A"),
                "job_requirements": job_info.get("requirements", ""),
                "context": state["context"],
                "feedback_instruction": feedback_instruction
            })
            return {"analysis": structured_result.model_dump_json(), "revision_count": current_count + 1}
        except Exception as e:
            # Error Handling Constraint: Return default "No Match" object on failure
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
        reflection_prompt = PromptTemplate.from_template(
            "You are an auditing algorithm. Review the generated JSON analysis against the resume context. "
            "If the analysis claims a critical skill or soft skill is MISSING, but it ACTUALLY EXISTS in the context, output: 'FAIL: You claimed [Skill] is missing, but it is in the context.' "
            "If the analysis claims a skill is MATCHED, but it is NOT in the context, output: 'FAIL: You hallucinated [Skill].' "
            "If the analysis is accurate based strictly on the text, output exactly: 'PASS'."
            "\n\nContext:\n{context}\n\nAnalysis:\n{analysis}"
        )
        chain = reflection_prompt | self.llm
        feedback_msg = chain.invoke({"context": state["context"], "analysis": state["analysis"]})
        
        # Handle different response types based on the underlying chat model wrapper
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
    # Example Config Overrides - You can map these to your config.json
    # PROVIDER = "ollama"       # Options: 'openai', 'ollama', 'vllm'
    # MODEL_NAME = "llama3"     # e.g., 'gpt-4o', 'llama3', 'mistral'
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