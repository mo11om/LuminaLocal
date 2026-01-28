Here is a significantly improved and detailed `README.md`. I have expanded it to include installation instructions, expected data formats, configuration details, and an explanation of the output logic based on the code provided in `utils.py` and `update_apply.py`.

---

# AI Resume Matcher & Gap Analyzer

## 📖 Overview

This project is a privacy-focused, local **Batch RAG (Retrieval-Augmented Generation)** pipeline designed to automate technical resume screening.

Instead of sending personal data to cloud APIs, this system runs locally using **Ollama** and **ChromaDB**. It indexes a PDF resume, performs semantic searches against a database of job descriptions, and provides a structured "Gap Analysis" detailing exactly why a candidate matches (or doesn't match) a role.

## 🏗 System Architecture
![Alt Text](./image.png)
The application is built on a modular "Agentic" workflow:

* **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) (Manages the state between retrieval and analysis nodes).
* **LLM Inference:** [Ollama](https://ollama.com/) (Runs local models like `gpt-oss`, `llama3`, or `mistral`).
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (Stores resume embeddings locally).
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) via `sentence-transformers`.
* **Validation:** Pydantic (Enforces strict JSON schema for the AI output).

---

## ⚙️ Prerequisites

1. **Python 3.10+**
2. **Ollama**: Download and install from [ollama.com](https://ollama.com).

### Model Setup

The system is configured to use `gpt-oss:20b` by default (defined in `utils.py`). You must pull this model before running the script. You can also swap this for `llama3` or `mistral` if your hardware is constrained.

```bash
# Pull the default model
ollama pull gpt-oss:20b

# OR, if you change the config to llama3
ollama pull llama3

```

---

## 🚀 Installation & Setup

1. **Clone the Repository**
```bash
git clone <your-repo-url>
cd <your-repo-folder>

```


2. **Install Dependencies**
Based on the imports, install the required packages:
```bash
pip install langchain langchain-community langchain-ollama langchain-huggingface langgraph chromadb pypdf pydantic

```


3. **Directory Structure**
Ensure your folders look like this (or update `utils.py` to match your paths):
```text
.
├── utils.py           # Configuration & Helper classes
├── update_apply.py    # Main execution script
├── cv/
│   └── cv_ver2.pdf    # Your Resume PDF
├── jobs/
│   └── processed_jobs_schema_all.json  # Input Jobs JSON
├── output/            # Generated CSV results appear here
└── chroma_db/         # Auto-generated Vector Database

```



---

## 📝 Configuration

All settings are centralized in `utils.py` under the `Config` class. You can modify these values to change the model, file paths, or tuning parameters.

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_MODEL` | `gpt-oss:20b` | The Ollama model tag to use. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | The HuggingFace model for vectorizing text. |
| `RESUME_FILE` | `./cv/cv_ver2.pdf` | Path to the candidate's PDF resume. |
| `JOBS_FILE` | `./jobs/...json` | Path to the JSON file containing job listings. |
| `CHUNK_SIZE` | `500` | Character limit for splitting the resume. |

---

## 📂 Data Preparation

### 1. The Resume

Place your PDF resume in the `cv/` folder. Ensure the filename matches `Config.RESUME_FILE` in `utils.py`.

### 2. The Jobs JSON

The system expects a JSON list of job objects in the `jobs/` folder. The logic specifically looks for `job_title` and `requirements` (or `description`) fields.

**Example JSON format:**

```json
[
  {
    "job_title": "Senior Python Developer",
    "department": "Engineering",
    "requirements": "Must have 5+ years of Python, Django, and AWS experience..."
  },
  {
    "job_title": "Frontend Engineer",
    "department": "UI/UX",
    "requirements": "Experience with React, TypeScript, and Tailwind..."
  }
]

```

---

## 🏃 Usage

Run the main script to process the batch:

```bash
python update_apply.py

```

### What happens during execution?

1. **Ingestion:** The script checks if the resume PDF exists. It splits the PDF into chunks and embeds them into a local ChromaDB instance.
2. **Retrieval:** For every job in your JSON file, the system queries the Vector DB to find specific parts of your resume relevant to that job.
3. **Analysis:** The LLM acts as a "Technical Recruiter" to compare the retrieved resume context against the job requirements.
4. **Export:** Results are printed to the console and saved to `output/analysis_results.csv`.

---

## 📊 Output Explanation

The results are saved to `analysis_results.csv`. The AI classifies the fit into four categories:

* **High Match:** Candidate possesses **90%+** of critical skills.
* **Medium Match:** Candidate has **60-90%** of skills or strong transferrable knowledge.
* **Low Match:** Missing significant core technologies (e.g., Job needs Java, Resume only has Python).
* **No Match:** Irrelevant background.

### CSV Columns

* **Missing Critical Skills:** Mandatory tech stacks found in the Job Description but not in the Resume.
* **Missing Bonus Skills:** Nice-to-have qualifications.
* **Keyword Optimization:** Suggestions on how to rename skills in the resume to pass ATS filters for this specific role.
* **Brief Analysis:** A 2-sentence summary of the decision.

---

## 🛠 Troubleshooting

* **`ConnectionRefusedError`**: Ensure Ollama is running in the background (`ollama serve`).
* **`FileNotFoundError`**: Check that your `cv/` and `jobs/` folders exist and contain the correct files defined in `utils.py`.
* **JSON Errors**: If the analysis fails for a specific job, the script will default to "No Match" and continue to the next job to prevent crashing. Check the console output for specific error messages.