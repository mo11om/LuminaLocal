import os
from typing import List
from utils import Config, ResumeIngestor, BaseAgentState

# ==========================================
# UPDATED IMPORT: LangChain Ollama
# ==========================================
from langchain_ollama import OllamaLLM  # <--- FIXED

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ==========================================
# Define the Output Schema
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
# LangGraph Workflow
# ==========================================
class ResumeGraphBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        # UPDATED: Use OllamaLLM instead of Ollama
        self.llm = OllamaLLM(model=Config.LLM_MODEL) # <--- FIXED
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

    def retrieve_node(self, state: BaseAgentState):
        print("🔍 Node: Retrieving relevant resume parts...")
        question = state["job_description"]
        documents = self.retriever.invoke(question)
        context_str = "\n\n".join([doc.page_content for doc in documents])
        return {"context": context_str}

    def analyze_node(self, state: BaseAgentState):
        print("🤖 Node: Analyzing fit with LLM (Structured Output)...")
        parser = PydanticOutputParser(pydantic_object=AnalysisResult)

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

        chain = prompt | self.llm | parser
        
        try:
            structured_result = chain.invoke({
                "job_description": state["job_description"], 
                "context": state["context"]
            })
            return {"analysis": structured_result.model_dump_json(indent=2)}
        except Exception as e:
            print(f"❌ Error parsing output: {e}")
            return {"analysis": f"Error: {str(e)}"}
            
    def build(self):
        workflow = StateGraph(BaseAgentState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", END)
        return workflow.compile()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    dummy_pdf_path = Config.RESUME_FILE 

    if os.path.exists(dummy_pdf_path):
        ingestor = ResumeIngestor()
        vector_store = ingestor.get_vector_store(dummy_pdf_path)

        graph_builder = ResumeGraphBuilder(vector_store)
        app = graph_builder.build()

        target_jd = """
        Looking for a Senior Python Developer with:
        - Strong LangGraph and LangChain experience.
        - Knowledge of RAG systems.
        - Experience with Docker and Kubernetes.
        """

        print("\n🚀 Starting LangGraph Workflow...")
        inputs = {"job_description": target_jd}
        
        result = app.invoke(inputs)

        print("\n" + "="*50)
        print("📊 Final Analysis Result")
        print("="*50)
        print(result["analysis"])
        
    else:
        print(f"❌ Please place a resume PDF at '{dummy_pdf_path}' to run.")