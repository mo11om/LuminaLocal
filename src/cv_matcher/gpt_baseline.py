"""
GPT Baseline Module: Provides OpenAI GPT-4o baseline comparison for resume matching.
This module complements the local LLM agent by providing an external LLM comparison.
"""

import os
import json
from typing import Dict, Literal, List
from openai import OpenAI, OpenAIError
from .utils import Config

# ==========================================
# GPT Baseline Output Schema
# ==========================================
class GPTBaselineResult:
    """
    Structured result from GPT baseline analysis.
    """
    def __init__(self, 
                 match_classification: Literal["High Match", "Medium Match", "Low Match", "No Match"],
                 missing_critical_skills: List[str],
                 brief_analysis: str):
        self.match_classification = match_classification
        self.missing_critical_skills = missing_critical_skills
        self.brief_analysis = brief_analysis

    def to_dict(self) -> Dict:
        """Convert result to dictionary format."""
        return {
            "match_classification": self.match_classification,
            "missing_critical_skills": self.missing_critical_skills,
            "brief_analysis": self.brief_analysis
        }

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# ==========================================
# GPT Baseline Function
# ==========================================
def get_gpt_baseline(job_title: str, requirements: str, resume_context: str) -> Dict:
    """
    Call OpenAI's GPT-4o to provide a baseline comparison for resume matching.
    
    Args:
        job_title: The job position title
        requirements: The job requirements/description text
        resume_context: The retrieved resume context relevant to the job
    
    Returns:
        Dictionary with match_classification, missing_critical_skills, and brief_analysis.
        On error, returns a structured error response.
    """
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY not found in environment. Skipping GPT baseline.")
        return {
            "error": "OPENAI_API_KEY not configured",
            "match_classification": "No Match",
            "missing_critical_skills": [],
            "brief_analysis": "Cannot perform GPT baseline analysis without API key."
        }
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # System prompt: Technical recruiting assistant
        system_prompt = """You are a Principal Staff Engineer acting as a Technical Recruiter.
Your goal is to classify a candidate's fit for a specific role based on their resume and the job requirements.

CLASSIFICATION RULES:
1. **High Match**: Candidate possesses 90%+ of the "Must-Have" technical skills found in the JD.
2. **Medium Match**: Candidate possesses 60-90% of "Must-Have" skills or has strong transferrable skills.
3. **Low Match**: Candidate lacks significant core technologies required (e.g., JD needs Java, Resume only has Python).
4. **No Match**: Irrelevant background or missing core competencies.

Analyze objectively and carefully. Do not hallucinate skills not present in the RESUME CONTEXT.

You MUST respond with ONLY valid JSON (no markdown, no explanations before or after).
The JSON must contain exactly these fields:
- match_classification: One of "High Match", "Medium Match", "Low Match", or "No Match"
- missing_critical_skills: Array of mandatory skills from JD missing in the resume
- brief_analysis: A concise 2-sentence summary of the classification"""

        # User prompt: Gap analysis request
        user_prompt = f"""### TARGET JOB
Title: {job_title}
Requirements:
{requirements}

### CANDIDATE RESUME CONTEXT
{resume_context}

### INSTRUCTIONS
Perform a Gap Analysis and provide your response as valid JSON ONLY. 
No additional text before or after the JSON."""

        # Call GPT-4o
        response = client.chat.completions.create(
            model=Config.GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=Config.GPT_TEMPERATURE,
            max_tokens=Config.GPT_MAX_TOKENS,
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        try:
            result_json = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response if wrapped in markdown
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                result_json = json.loads(response_text[json_start:json_end])
            else:
                raise ValueError("Failed to parse GPT response as JSON")
        
        # Validate required fields
        required_fields = ["match_classification", "missing_critical_skills", "brief_analysis"]
        for field in required_fields:
            if field not in result_json:
                raise ValueError(f"Missing required field in GPT response: {field}")
        
        # Validate match classification
        valid_classifications = ["High Match", "Medium Match", "Low Match", "No Match"]
        if result_json["match_classification"] not in valid_classifications:
            raise ValueError(f"Invalid classification: {result_json['match_classification']}")
        
        # Ensure missing_critical_skills is a list
        if not isinstance(result_json.get("missing_critical_skills"), list):
            result_json["missing_critical_skills"] = []
        
        return {
            "match_classification": result_json["match_classification"],
            "missing_critical_skills": result_json["missing_critical_skills"],
            "brief_analysis": result_json["brief_analysis"]
        }
    
    except OpenAIError as e:
        print(f"⚠️ OpenAI API Error: {str(e)}")
        return {
            "error": f"OpenAI API Error: {str(e)}",
            "match_classification": "No Match",
            "missing_critical_skills": [],
            "brief_analysis": f"Error communicating with OpenAI API: {str(e)}"
        }
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse GPT response as JSON: {str(e)}")
        return {
            "error": f"JSON parsing error: {str(e)}",
            "match_classification": "No Match",
            "missing_critical_skills": [],
            "brief_analysis": "Error parsing GPT response. Could not extract structured data."
        }
    except Exception as e:
        print(f"⚠️ Unexpected error in GPT baseline: {str(e)}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "match_classification": "No Match",
            "missing_critical_skills": [],
            "brief_analysis": f"Unexpected error during baseline analysis: {str(e)}"
        }