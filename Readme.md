# AI Resume Matcher & Gap Analyzer

## 📖 Overview

This project is a privacy-focused, local **Batch RAG (Retrieval-Augmented Generation)** pipeline designed to automate technical resume screening.

Instead of sending personal data to cloud APIs, this system runs locally using **Ollama** and **ChromaDB**. It indexes a PDF resume, performs semantic searches against a database of job descriptions, and provides a structured "Gap Analysis" detailing exactly why a candidate matches (or doesn't match) a role.

## 🏗 System Architecture
![System Architecture Diagram](./image.png)

The application is built on a modular "Agentic" workflow:

* **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) (Manages the state between retrieval and analysis nodes).
* **LLM Inference:** [Ollama](https://ollama.com/) (Runs local models like `gpt-oss`, `llama3`, or `mistral`).
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (Stores resume embeddings locally).
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) via `sentence-transformers`.
* **Validation:** Pydantic (Enforces strict JSON schema for the AI output).

---

## 📂 Project Structure

```text
.
├── config.json               # All configuration (paths, models, hyperparameters)
├── example.json              # Example job JSON for testing
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment spec
├── image.png                 # Architecture diagram
│
├── src/
│   └── cv_matcher/           # Main Python package
│       ├── __init__.py
│       ├── utils.py          # Config class, ResumeIngestor, shared types
│       ├── main.py           # Primary batch analysis script (use this to run)
│       ├── agent.py          # Advanced agentic workflow with reflection loop
│       ├── update_cv.py      # Single-job gap analysis utility
│       └── cv_ai.py          # Standalone prototype / legacy script
│
└── data/
    ├── raw/                  # Your resume PDF(s)
    ├── jobs/                 # Job JSON files
    ├── output/               # Generated CSV reports
    └── chroma_db/            # Auto-generated vector database
```

---

## ⚙️ Prerequisites

1. **Python 3.10+**
2. **Ollama**: Download and install from [ollama.com](https://ollama.com).

### Model Setup

The system is configured to use `gpt-oss:20b` by default (defined in `config.json`). You must pull this model before running the script.

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
   ```bash
   pip install -r requirements.txt
   ```

   Or with conda:
   ```bash
   conda env create -f environment.yml
   conda activate <env-name>
   ```

3. **Install the Package (required for `src/` layout)**
   ```bash
   pip install -e .
   ```
   This registers `cv_matcher` on your Python path so `python -m cv_matcher.main` works.

4. **Place Your Files**
   - Copy your **resume PDF** to `data/raw/` and update `RESUME_FILE` in `config.json`.
   - Copy your **jobs JSON** to `data/jobs/` and update `JOBS_FILE` in `config.json`.

---

## ⚙️ Configuration

All settings are centralized in `config.json` at the project root.

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_MODEL` | `gpt-oss:20b` | The Ollama model tag to use. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model for vectorizing text. |
| `RESUME_FILE` | `./data/raw/CV.md.pdf` | Path to the candidate's PDF resume. |
| `JOBS_FILE` | `./data/jobs/processed_jobs_schema.json` | Path to the JSON file containing job listings. |
| `VECTOR_DB_PATH` | `./data/chroma_db` | Path to persist the ChromaDB vector store. |
| `OUTPUT_DIR` | `./data/output` | Directory where CSV reports are saved. |
| `CHUNK_SIZE` | `500` | Character limit for splitting the resume. |
| `RETRIEVER_K` | `8` | Number of resume chunks to retrieve per query. |

---

## 📝 Data Preparation

### 1. The Resume

Place your PDF resume in `data/raw/`. Update `config.json` to match the filename:
```json
"RESUME_FILE": "./data/raw/your_resume.pdf"
```

### 2. The Jobs JSON

The system expects a JSON list of job objects in `data/jobs/`. The logic specifically looks for `job_title` and `requirements` (or `description`) fields.

See `example.json` for the expected format:

```json
[
  {
    "job_title": "Senior Python Developer",
    "department": "Engineering",
    "requirements": "Must have 5+ years of Python, Django, and AWS experience..."
  }
]
```

---

## 🏃 Usage

Run the main batch analysis script from the **project root**:

```bash
python -m cv_matcher.main
```

Or, for the advanced agentic workflow with a reflection/self-correction loop:

```bash
python -m cv_matcher.agent
```

### What happens during execution?

1. **Ingestion:** Loads and chunks the resume PDF into ChromaDB.
2. **Retrieval:** For every job in the JSON, queries the vector store for relevant resume context.
3. **Analysis:** The LLM compares the context against job requirements and returns a structured JSON result.
4. **Export:** Results are sorted and saved to `data/output/analysis_results.csv`.

---

## 📊 Output Explanation

The results are saved to `data/output/analysis_results.csv`. The AI classifies fit into four categories:

| Classification | Criteria |
| --- | --- |
| **High Match** | Candidate possesses **90%+** of critical skills |
| **Medium Match** | Candidate has **60–90%** of skills or strong transferrable knowledge |
| **Low Match** | Missing significant core technologies |
| **No Match** | Irrelevant background |

### CSV Columns

* **Missing Critical Skills:** Mandatory tech stacks found in the JD but not in the Resume.
* **Missing Bonus Skills:** Nice-to-have qualifications.
* **Keyword Optimization:** Suggestions on how to rename skills to pass ATS filters.
* **Brief Analysis:** A 2-sentence summary of the decision.

---

## 🛠 Troubleshooting

* **`ConnectionRefusedError`**: Ensure Ollama is running (`ollama serve`).
* **`FileNotFoundError`**: Verify that `data/raw/` and `data/jobs/` exist and contain the files defined in `config.json`.
* **`⚠️ Config file not found`**: Always run scripts from the **project root** so that `config.json` is found correctly.
* **JSON Errors**: If analysis fails for a specific job, the script defaults to "No Match" and continues. Check console output for details.