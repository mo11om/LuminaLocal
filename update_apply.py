import os
import json
import csv
from typing import List, Literal
from utils import Config, ResumeIngestor, BaseAgentState

# ==========================================
# UPDATED IMPORT: LangChain Ollama
# ==========================================
from langchain_ollama import OllamaLLM 

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ==========================================
# Output Schema
# ==========================================
class AnalysisResult(BaseModel):
    match_classification: Literal["High Match", "Medium Match", "Low Match", "No Match"] = Field(
        description="The categorical fit. 'High Match': 90%+ critical skills. 'Medium Match': 60-90%. 'Low Match': <60%."
    )
    missing_critical_skills: List[str] = Field(
        description="List of mandatory technical skills found in the JD but strictly missing in the resume."
    )
    missing_bonus_skills: List[str] = Field(
        description="List of nice-to-have skills or 'bonus' qualifications missing from the resume."
    )
    keyword_optimization_suggestions: List[str] = Field(
        description="Actionable advice on renaming skills or adding specific keywords to pass ATS."
    )
    brief_analysis: str = Field(
        description="A concise, 2-sentence summary of why this classification was assigned."
    )

# ==========================================
# LangGraph Workflow
# ==========================================
class ResumeGraphBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = OllamaLLM(model=Config.LLM_MODEL)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})

    def retrieve_node(self, state: BaseAgentState):
        question = state["job_description"]
        documents = self.retriever.invoke(question)
        context_str = "\n\n".join([doc.page_content for doc in documents])
        return {"context": context_str}

    def analyze_node(self, state: BaseAgentState):
        parser = PydanticOutputParser(pydantic_object=AnalysisResult)

        CLASSIFICATION_TEMPLATE = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a Principal Staff Engineer acting as a Technical Recruiter.
        Your goal is to classify the candidate's fit for a specific role based strictly on the provided context.
        
        {format_instructions}
        
        ### CLASSIFICATION RULES:
        1. **High Match**: Candidate possesses 90%+ of the "Must-Have" technical skills found in the JD.
        2. **Medium Match**: Candidate possesses 60-80% of "Must-Have" skills or has strong transferrable skills.
        3. **Low Match**: Candidate lacks significant core technologies required (e.g., JD needs Java, Resume only has Python).
        4. **No Match**: Irrelevant background.

        Analyze objectively. Do not hallucinate skills not present in the RESUME CONTEXT.
        <|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        ### TARGET JOB
        Title: {job_title}
        Department: {job_department}
        Requirements:
        {job_requirements}

        ### CANDIDATE RESUME CONTEXT
        {context}

        ### INSTRUCTIONS
        Perform the Gap Analysis and output the JSON result.
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

        prompt = PromptTemplate(
            template=CLASSIFICATION_TEMPLATE,
            input_variables=["job_title", "job_department", "job_requirements", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.llm | parser
        
        job_info = state.get("job_info", {})

        try:
            structured_result = chain.invoke({
                "job_title": job_info.get("job_title", "N/A"),
                "job_department": job_info.get("department", "N/A"),
                "job_requirements": job_info.get("requirements", ""),
                "context": state["context"]
            })
            return {"analysis": structured_result.model_dump_json()}
            
        except Exception as e:
            # Return a fallback JSON on error so the pipeline continues
            return {"analysis": json.dumps({"error": str(e), "match_classification": "No Match"})}

    def build(self):
        workflow = StateGraph(BaseAgentState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", END)
        return workflow.compile()

# ==========================================
# Main Execution (Batch)
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
    
    graph_builder = ResumeGraphBuilder(vector_store)
    app = graph_builder.build()

    print(f"\n📂 Loading Job Descriptions from {Config.JOBS_FILE}...")
    with open(Config.JOBS_FILE, 'r', encoding='utf-8') as f:
        job_list = json.load(f)

    results_summary = []

    print(f"🚀 Starting Batch Analysis for {len(job_list)} jobs...\n")

    for i, job in enumerate(job_list):
        jd_query = f"{job.get('job_title', '')} {job.get('requirements', '')}"
        
        inputs = {
            "job_info": job,
            "job_description": jd_query, 
            "context": "",
            "analysis": ""
        }

        try:
            output = app.invoke(inputs)
            result_json = json.loads(output["analysis"])
            
            title = job.get('job_title')[:50]
            classification = result_json.get('match_classification', 'Error')
            
            print(f"[{i+1}/{len(job_list)}] {classification.upper()} : {title}...")
            
            # === CHANGED: Removed the 'if' filter to keep ALL results ===
            results_summary.append({
                "job_title": job.get('job_title'),
                "classification": classification,
                "analysis": result_json
            })
                
        except Exception as e:
            print(f"[{i+1}] ❌ Failed to process: {e}")

    print("\n" + "="*60)
    print("📊 CANDIDATE SUITABILITY REPORT")
    print("="*60)
    
    if not results_summary:
        print("No results generated.")
    else:
        # Sort logic: High -> Medium -> Low -> No Match -> Others
        sort_order = {
            "High Match": 1,
            "Medium Match": 2,
            "Low Match": 3,
            "No Match": 4
        }
        results_summary.sort(key=lambda x: sort_order.get(x['classification'], 5))
        
        # Print summary to console
        for res in results_summary:
            print(f"\n🔹 [{res['classification']}] {res['job_title']}")
            # Use .get() safely in case of 'No Match' errors lacking these fields
            print(f"   Missing Critical: {res['analysis'].get('missing_critical_skills', [])}")
            print(f"   Summary: {res['analysis'].get('brief_analysis', 'N/A')}")

        # ==========================================
        # CSV EXPORT LOGIC
        # ==========================================
        csv_filename = "analysis_results.csv"
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
                    'Missing Critical Skills', 
                    'Missing Bonus Skills', 
                    'Keyword Optimization', 
                    'Brief Analysis'
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for res in results_summary:
                    analysis = res['analysis']
                    
                    # Helper to flatten lists safely (handles missing keys for Low/No match)
                    def flatten_list(lst):
                        if not lst: return ""
                        return ", ".join(lst) if isinstance(lst, list) else str(lst)

                    writer.writerow({
                        'Job Title': res['job_title'],
                        'Classification': res['classification'],
                        'Missing Critical Skills': flatten_list(analysis.get('missing_critical_skills')),
                        'Missing Bonus Skills': flatten_list(analysis.get('missing_bonus_skills')),
                        'Keyword Optimization': flatten_list(analysis.get('keyword_optimization_suggestions')),
                        'Brief Analysis': analysis.get('brief_analysis', '')
                    })
            print("✅ CSV export complete.")
            
        except Exception as e:
            print(f"❌ Error writing CSV: {e}")

    print("\n✅ Done.")