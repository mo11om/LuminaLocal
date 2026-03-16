import os
import json
import csv
from typing import List, Literal, TypedDict, Optional, Dict, Any
from .utils import Config, ResumeIngestor, BaseAgentState

from langchain_ollama import OllamaLLM 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ==========================================
# 1. Output Schema
# ==========================================
class AnalysisResult(BaseModel):
    match_classification: Literal["Strong Match", "Good Match", "No Match"] = Field(
        description="The categorical fit. 'Strong Match': 70%+ critical skills. 'Good Match': 40-70%. 'No Match': <40% or irrelevant."
    )
    matching_skills: List[str] = Field(
        description="List of key technical skills found in BOTH the Job Description and the Candidate's Resume."
    )
    missing_critical_skills: List[str] = Field(
        description="List of mandatory technical skills found in the JD but strictly missing in the resume."
    )
    missing_bonus_skills: List[str] = Field(
        description="List of nice-to-have skills or 'bonus' qualifications missing from the resume."
    )

# ==========================================
# 2. Agentic State (extends BaseAgentState with agentic fields)
# ==========================================
class AgenticState(BaseAgentState, total=False):
    optimized_query: str
    feedback: str
    revision_count: int

# ==========================================
# 3. Agentic LangGraph Workflow
# ==========================================
class AdvancedResumeGraphBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = OllamaLLM(model=Config.LLM_MODEL)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})

    def transform_query_node(self, state: AgenticState):
        print("🪄  Node: Transforming JD into optimized search query...")
        prompt = PromptTemplate.from_template(
            "Extract a comma-separated list of the core programming languages, frameworks, "
            "and tools from this job description. Do not include soft skills. JD: {jd}"
        )
        chain = prompt | self.llm
        optimized_query = chain.invoke({"jd": state["job_description"]})
        return {"optimized_query": optimized_query.strip()}

    def retrieve_node(self, state: AgenticState):
        print("🔍 Node: Retrieving context using optimized query...")
        documents = self.retriever.invoke(state.get("optimized_query", state["job_description"]))
        context_str = "\n\n".join([doc.page_content for doc in documents])
        return {"context": context_str}

    def analyze_node(self, state: AgenticState):
        current_count = state.get("revision_count", 0)
        print(f"🤖 Node: Generating Analysis (Draft {current_count + 1})...")
        
        parser = PydanticOutputParser(pydantic_object=AnalysisResult)
        
        # Inject feedback if the agent is correcting a hallucination
        feedback_instruction = ""
        if state.get("feedback") and "FAIL" in state.get("feedback", ""):
            feedback_instruction = f"\n### CRITICAL FEEDBACK FROM PREVIOUS DRAFT:\n{state['feedback']}\nYou MUST fix these errors in your new JSON output."

        CLASSIFICATION_TEMPLATE = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a Principal Staff Engineer acting as a Technical Recruiter.
        Your goal is to classify the candidate's fit for a specific role based strictly on the provided context.
        
        {format_instructions}
        {feedback_instruction}
        
        ### CLASSIFICATION RULES:
        1. **Strong Match**: Candidate possesses 70%+ of the "Must-Have" technical skills found in the JD.
        2. **Good Match**: Candidate possesses 40-70% of "Must-Have" skills or has strong transferrable skills.
        3. **No Match**: Candidate lacks significant core technologies required (<40% match) or has a completely irrelevant background.

        Analyze objectively. Extract the matching skills, missing critical skills, and missing bonus skills. Do not hallucinate skills not present in the RESUME CONTEXT.
        <|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        ### TARGET JOB
        Title: {job_title}
        Department: {job_department}
        Requirements: {job_requirements}

        ### CANDIDATE RESUME CONTEXT
        {context}

        ### INSTRUCTIONS
        Perform the Gap Analysis and output the JSON result.
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

        prompt = PromptTemplate(
            template=CLASSIFICATION_TEMPLATE,
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
            return {"analysis": json.dumps({"error": str(e), "match_classification": "No Match"}), "revision_count": current_count + 1}

    def reflect_node(self, state: AgenticState):
        print("🧐 Node: Reflecting and Validating Output...")
        reflection_prompt = PromptTemplate.from_template(
            "You are an auditing algorithm. Review the generated JSON analysis against the resume context. "
            "If the analysis claims a skill is missing but it ACTUALLY EXISTS in the context, output: 'FAIL: You claimed X is missing, but it is in the context.' "
            "If the analysis is accurate, output exactly: 'PASS'."
            "\n\nContext:\n{context}\n\nAnalysis:\n{analysis}"
        )
        chain = reflection_prompt | self.llm
        feedback = chain.invoke({
            "context": state["context"], 
            "analysis": state["analysis"]
        })
        print(f"   -> Result: {feedback.strip()}")
        return {"feedback": feedback.strip()}

    def should_reanalyze(self, state: AgenticState) -> str:
        if state.get("revision_count", 0) >= 3:
            return END
        if "FAIL" in state.get("feedback", ""):
            print("🔄 Re-routing to Analyze Node to fix hallucinations...")
            return "analyze"
        return END

    def build(self):
        workflow = StateGraph(AgenticState)
        workflow.add_node("transform_query", self.transform_query_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("reflect", self.reflect_node)
        
        workflow.set_entry_point("transform_query")
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", "reflect")
        workflow.add_conditional_edges("reflect", self.should_reanalyze, {"analyze": "analyze", END: END})
        
        return workflow.compile()

# ==========================================
# 4. Main Execution (Batch)
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(Config.RESUME_FILE):
        print(f"❌ Error: Resume file '{Config.RESUME_FILE}' not found.")
        exit(1)

    if not os.path.exists(Config.JOBS_FILE):
        print(f"❌ Error: Jobs file '{Config.JOBS_FILE}' not found.")
        exit(1)

    ingestor = ResumeIngestor()
    vector_store = ingestor.get_vector_store(Config.RESUME_FILE)
    
    app = AdvancedResumeGraphBuilder(vector_store).build()

    print(f"\n📂 Loading Job Descriptions from {Config.JOBS_FILE}...")
    with open(Config.JOBS_FILE, 'r', encoding='utf-8') as f:
        job_list = json.load(f)

    results_summary = []
    print(f"🚀 Starting AGENTIC Batch Analysis for {len(job_list)} jobs...\n")

    for i, job in enumerate(job_list):
        jd_query = f"{job.get('job_title', '')} {job.get('requirements', '')}"
        inputs = {
            "job_info": job,
            "job_description": jd_query,
            "context": "",
            "revision_count": 0
        }

        try:
            print(f"\n--- Processing Job {i+1}/{len(job_list)}: {job.get('job_title')} ---")
            output = app.invoke(inputs)
            result_json = json.loads(output["analysis"])

            title = job.get('job_title', '')[:50]
            classification = result_json.get('match_classification', 'Error')
            print(f"[{i+1}/{len(job_list)}] {classification.upper()} : {title}...")
            
            results_summary.append({
                "job_title": job.get('job_title'),
                "classification": classification,
                "analysis": result_json
            })
        except Exception as e:
            print(f"[{i+1}] ❌ Failed: {e}")

    print("\n" + "="*60)
    print("📊 CANDIDATE SUITABILITY REPORT")
    print("="*60)

    if not results_summary:
        print("No results generated.")
    else:
        sort_order = {"Strong Match": 1, "Good Match": 2, "No Match": 3}
        results_summary.sort(key=lambda x: sort_order.get(x['classification'], 4))

        for res in results_summary:
            print(f"\n🔹 [{res['classification']}] {res['job_title']}")
            print(f"   Matching Skills  : {res['analysis'].get('matching_skills', [])}")
            print(f"   Missing Critical : {res['analysis'].get('missing_critical_skills', [])}")

        # ==========================================
        # CSV EXPORT LOGIC
        # ==========================================
        csv_filename = "agentic_" + Config.ANALYSIS_OUTPUT_CSV
        output_dir = getattr(Config, 'OUTPUT_DIR', '.')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        csv_path = os.path.join(output_dir, csv_filename)
        print(f"\n💾 Saving all {len(results_summary)} results to CSV: {csv_path}")

        try:
            with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = [
                    'Job Title',
                    'Classification',
                    'Matching Skills',
                    'Missing Critical Skills',
                    'Missing Bonus Skills'
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for res in results_summary:
                    analysis = res['analysis']

                    def flatten_list(lst):
                        if not lst: return ""
                        return ", ".join(lst) if isinstance(lst, list) else str(lst)

                    writer.writerow({
                        'Job Title': res['job_title'],
                        'Classification': res['classification'],
                        'Matching Skills': flatten_list(analysis.get('matching_skills')),
                        'Missing Critical Skills': flatten_list(analysis.get('missing_critical_skills')),
                        'Missing Bonus Skills': flatten_list(analysis.get('missing_bonus_skills'))
                    })
            print("✅ CSV export complete.")
        except Exception as e:
            print(f"❌ Error writing CSV: {e}")

    print("\n✅ Done.")