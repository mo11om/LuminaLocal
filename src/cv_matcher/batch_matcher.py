"""
Batch Matcher Module: Single-call GPT strategy.
Sends ALL job descriptions in one prompt and parses a JSON array response.
"""

import os
import json
import csv
import re
from typing import List, Dict, Any
from .utils import Config, ResumeIngestor
from openai import OpenAI, OpenAIError


# ==========================================
# 1. Prompt Messages (system + user)
# ==========================================
SYSTEM_PROMPT = """You are a Principal Staff Engineer acting as a Technical Recruiter.
Your task is to evaluate a SINGLE candidate's resume against MULTIPLE Job Descriptions (JDs) at once.

### CLASSIFICATION RULES:
1. **Strong Match**: Candidate possesses 70%+ of the "Must-Have" technical skills found in the JD.
2. **Good Match**: Candidate possesses 40-70% of "Must-Have" skills or has strong transferrable skills.
3. **No Match**: Candidate lacks significant core technologies required (e.g., <40% match) or has a completely irrelevant background.

### INSTRUCTIONS:
- Analyze the resume objectively against EACH job description provided.
- Do not hallucinate skills not present in the RESUME CONTEXT.
- You MUST respond with ONLY a valid JSON array containing one object for each job evaluated. Do not wrap the JSON in markdown blocks or provide conversational text.

### OUTPUT SCHEMA (For each job object):
{
  "job_title": "<String: The title of the job>",
  "match_classification": "<String: 'Strong Match', 'Good Match', or 'No Match'>",
  "matched_critical_skills": ["<Array: 'Must-Have' JD skills PRESENT in the resume>"],
  "matched_bonus_skills": ["<Array: 'Nice-to-Have' JD skills PRESENT in the resume>"],
  "missing_critical_skills": ["<Array: 'Must-Have' JD skills MISSING in the resume>"],
  "missing_bonus_skills": ["<Array: 'Nice-to-Have' JD skills MISSING in the resume>"]
}

### FEW-SHOT EXAMPLES:

**EXAMPLE INPUT:**
<RESUME_CONTEXT>
Senior Python Developer with 5 years of experience building scalable backend systems.
Expertise in Python, Django, FastAPI, and PostgreSQL.
Deployed applications using Docker and AWS (EC2, S3). Familiar with basic CI/CD pipelines.
</RESUME_CONTEXT>

<JOB_DESCRIPTIONS>
[Job 1] Backend Engineer (Python)
Requirements: Must have strong Python, Django, and SQL database experience (PostgreSQL preferred). Nice to have: Docker, Kubernetes, AWS.

[Job 2] Lead DevOps Engineer
Requirements: Must have deep expertise in Kubernetes, Terraform, and advanced AWS networking. Nice to have: Python scripting, CI/CD setup.
</JOB_DESCRIPTIONS>

**EXAMPLE OUTPUT:**
[
  {
    "job_title": "Backend Engineer (Python)",
    "match_classification": "Strong Match",
    "matched_critical_skills": ["Python", "Django", "PostgreSQL"],
    "matched_bonus_skills": ["Docker", "AWS"],
    "missing_critical_skills": [],
    "missing_bonus_skills": ["Kubernetes"]
  },
  {
    "job_title": "Lead DevOps Engineer",
    "match_classification": "No Match",
    "matched_critical_skills": [],
    "matched_bonus_skills": ["Python", "CI/CD"],
    "missing_critical_skills": ["Kubernetes", "Terraform", "Advanced AWS networking"],
    "missing_bonus_skills": []
  }
]"""

USER_PROMPT_TEMPLATE = """### ACTUAL INPUT:

<RESUME_CONTEXT>
{resume_context}
</RESUME_CONTEXT>

<JOB_DESCRIPTIONS>
{job_descriptions_formatted_list}
</JOB_DESCRIPTIONS>

### YOUR JSON RESPONSE ONLY:"""


# ==========================================
# 2. Batch Resume Analyzer
# ==========================================
class BatchResumeAnalyzer:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment. Set it before running.")
        self.client = OpenAI(api_key=api_key)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})

    def _format_jobs(self, job_list: List[Dict[str, Any]]) -> str:
        """Format all jobs into a numbered list for the prompt."""
        lines = []
        for i, job in enumerate(job_list, 1):
            title = job.get("job_title", "Unknown")
            dept = job.get("department", "N/A")
            reqs = job.get("requirements", "N/A")
            lines.append(f"[Job {i}] {title}\nDepartment: {dept}\nRequirements: {reqs}")
        return "\n\n".join(lines)

    def _retrieve_context(self, job_list: List[Dict[str, Any]]) -> str:
        """Build a single retrieval query from all jobs and fetch context."""
        combined_query = " ".join(
            f"{j.get('job_title', '')} {j.get('requirements', '')}" for j in job_list
        )
        documents = self.retriever.invoke(combined_query)
        return "\n\n".join(doc.page_content for doc in documents)

    def _extract_json_array(self, raw_text: str) -> List[Dict]:
        """Robustly extract a JSON array from the LLM response."""
        text = raw_text.strip()

        # Strip markdown fences if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Find the outermost [ ... ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        return json.loads(text)

    def run(self, job_list: List[Dict[str, Any]]) -> List[Dict]:
        """Execute the batch analysis in a single GPT call."""
        print("🔍 Retrieving resume context...")
        resume_context = self._retrieve_context(job_list)

        print("📝 Formatting job descriptions...")
        jd_block = self._format_jobs(job_list)

        user_message = USER_PROMPT_TEMPLATE.format(
            resume_context=resume_context,
            job_descriptions_formatted_list=jd_block,
        )

        print(f"🤖 Calling GPT ({Config.GPT_MODEL}) with {len(job_list)} jobs in ONE batch...")
        try:
            response = self.client.chat.completions.create(
                model=Config.GPT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=Config.GPT_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw_output = response.choices[0].message.content.strip()
        except OpenAIError as e:
            print(f"❌ OpenAI API error: {e}")
            return []

        print("✅ GPT response received. Parsing JSON array...")
        try:
            results = self._extract_json_array(raw_output)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Failed to parse GPT output as JSON array: {e}")
            print(f"   Raw output (first 500 chars): {raw_output[:500]}")
            results = []

        return results


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(Config.RESUME_FILE):
        print(f"❌ Error: Resume file '{Config.RESUME_FILE}' not found.")
        exit(1)

    if not os.path.exists(Config.JOBS_FILE):
        print(f"❌ Error: Jobs file '{Config.JOBS_FILE}' not found.")
        exit(1)

    # Ingest resume into vector store
    ingestor = ResumeIngestor()
    vector_store = ingestor.get_vector_store(Config.RESUME_FILE)

    # Load jobs
    print(f"\n📂 Loading Job Descriptions from {Config.JOBS_FILE}...")
    with open(Config.JOBS_FILE, "r", encoding="utf-8") as f:
        job_list = json.load(f)

    print(f"🚀 Starting BATCH Analysis for {len(job_list)} jobs (single LLM call)...\n")

    # Run batch
    analyzer = BatchResumeAnalyzer(vector_store)
    results = analyzer.run(job_list)

    # ==========================================
    # Console Report
    # ==========================================
    print("\n" + "=" * 60)
    print("📊 CANDIDATE SUITABILITY REPORT")
    print("=" * 60)

    if not results:
        print("No results generated.")
    else:
        sort_order = {"Strong Match": 1, "Good Match": 2, "No Match": 3}
        results.sort(key=lambda x: sort_order.get(x.get("match_classification", ""), 4))

        for res in results:
            print(f"\n🔹 [{res.get('match_classification', 'N/A')}] {res.get('job_title', 'Unknown')}")
            print(f"   Matched Critical : {res.get('matched_critical_skills', [])}")
            print(f"   Missing Critical : {res.get('missing_critical_skills', [])}")
            print(f"   Matched Bonus    : {res.get('matched_bonus_skills', [])}")
            print(f"   Missing Bonus    : {res.get('missing_bonus_skills', [])}")

        # ==========================================
        # CSV Export
        # ==========================================
        csv_filename = "batch_" + Config.ANALYSIS_OUTPUT_CSV
        output_dir = getattr(Config, "OUTPUT_DIR", ".")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        csv_path = os.path.join(output_dir, csv_filename)
        print(f"\n💾 Saving all {len(results)} results to CSV: {csv_path}")

        try:
            with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
                fieldnames = [
                    "Job Title",
                    "Classification",
                    "Matched Critical Skills",
                    "Missing Critical Skills",
                    "Matched Bonus Skills",
                    "Missing Bonus Skills",
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for res in results:
                    def flatten(lst):
                        if not lst:
                            return ""
                        return ", ".join(lst) if isinstance(lst, list) else str(lst)

                    writer.writerow({
                        "Job Title": res.get("job_title", ""),
                        "Classification": res.get("match_classification", ""),
                        "Matched Critical Skills": flatten(res.get("matched_critical_skills")),
                        "Missing Critical Skills": flatten(res.get("missing_critical_skills")),
                        "Matched Bonus Skills": flatten(res.get("matched_bonus_skills")),
                        "Missing Bonus Skills": flatten(res.get("missing_bonus_skills")),
                    })
            print("✅ CSV export complete.")
        except Exception as e:
            print(f"❌ Error writing CSV: {e}")

    print("\n✅ Done.")
