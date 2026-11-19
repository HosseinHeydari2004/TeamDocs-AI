questions = [
    # --- 1. Factual / Direct Retrieval ---
    "What is the purpose of TeamDocs AI?",
    "What chunk size and overlap are used during ingestion?",
    "Which vector database does the test configuration use?",
    "What is the maximum file size for uploaded documents?",
    "How should I name a branch for a new feature?",
    "What is the commit message format required by the team?",
    "How do I start the application locally?",
    "What is the minimum unit-test coverage required?",
    "What does the health endpoint return?",
    "How many chunks are retrieved per question by default?",

    # --- 2. Numeric / Precision ---
    "Which port is the application available on after starting it locally?",
    "How many stages are there in the RAG pipeline?",
    "What similarity score is shown for the README.md source in the sample query response?",
    "How many reviewers are required for each pull request?",
    "How many document formats are accepted by the upload endpoint?",

    # --- 3. API Reference / Entity Recall ---
    "What are the fields in the request body of POST /api/v1/query?",
    "What does the DELETE /api/v1/documents/{document_id} endpoint do?",
    "What HTTP method and path are used to check the service health?",
    "What does the sources array in the query response contain?",
    "What file formats are accepted by POST /api/v1/documents?",

    # --- 4. Multi-hop / Composite ---
    "What environment variables must be set in the .env file?",
    "What are the five stages of the RAG pipeline in order?",
    "How do I re-index documents after updating them?",
    "What code quality tools must be run before opening a pull request?",
    "What steps are required to run the project locally from scratch?",
    "Which technologies make up the test configuration stack?",

    # --- 5. Out-of-Scope / Hallucination Check ---
    "What is the company's vacation policy?",
    "What is the salary of a senior engineer?",
    "Who is the CEO of the company?",
    "Which version of Python is officially supported?",
    "How many developers are on the team?",

    # --- 6. Paraphrase / Semantic Robustness ---
    "Which vector store does the system rely on?",
    "What is the size of each text chunk during preprocessing?",
    "What branch naming convention should I follow for a bug fix?",
    "What happens if the answer is not present in the documents?",
    "What is the required test coverage for core modules?",

    # --- 7. Noise Sensitivity / Adversarial ---
    "Does TeamDocs AI use PostgreSQL as its vector database?",
    "Is the default chunk size 1000 tokens?",
    "Is committing directly to main allowed?",
    "Is the maximum upload size 50 MB?",
]

reference_answers = [
    # --- 1. Factual ---
    "TeamDocs AI is an AI-powered RAG assistant for software teams to search, retrieve, and understand internal documentation and technical knowledge.",
    "About 500 tokens with a 50-token overlap.",
    "ChromaDB.",
    "20 MB.",
    "feature/<name>",
    "Conventional Commits, e.g., feat: add PDF loader.",
    "Clone the repo, create a venv, install requirements, copy .env.example to .env, run docker compose up --build, then open http://localhost:8501.",
    "80% for core modules (ingestion, chunking, retrieval).",
    '{"status": "ok"}',
    "Top 5 chunks by default, configurable via the top_k parameter.",

    # --- 2. Numeric ---
    "Port 8501 (http://localhost:8501).",
    "Five stages: Ingestion, Chunking, Embedding, Retrieval, Generation.",
    "0.87.",
    "At least one reviewer, plus passing CI checks.",
    "Four formats: .md, .pdf, .docx, and .txt.",

    # --- 3. API / Entity ---
    "question, project, and top_k.",
    "It removes a document and all of its chunks from the vector database.",
    "GET /api/v1/health.",
    "Each source contains the file name, the section, and a score (e.g., README.md, Getting Started, 0.87).",
    "Accepted formats are .md, .pdf, .docx, and .txt.",

    # --- 4. Multi-hop ---
    "LLM_API_KEY and VECTOR_DB_PATH.",
    "Ingestion, Chunking, Embedding, Retrieval, Generation.",
    "Run the re-index action from the admin dashboard, or delete and re-upload the document through the API.",
    "Format Python code with Black and lint with Ruff before opening a pull request.",
    "Clone the repo, create a virtual environment, activate it, pip install -r requirements.txt, copy .env.example to .env and set the required variables, run docker compose up --build, then open http://localhost:8501.",
    "Python + FastAPI (backend), ChromaDB (vector DB), Sentence-Transformers multilingual (embeddings), a configurable LLM (local or hosted), Streamlit (frontend), and Docker Compose (deployment).",

    # --- 5. Out-of-Scope ---
    "Not found in the knowledge base.",
    "Not found in the knowledge base.",
    "Not found in the knowledge base.",
    "Not found in the knowledge base.",
    "Not found in the knowledge base.",

    # --- 6. Paraphrase ---
    "ChromaDB.",
    "About 500 tokens per chunk.",
    "fix/<name>",
    "The assistant should say it could not find the information in the knowledge base instead of guessing.",
    "80% unit-test coverage for core modules.",

    # --- 7. Noise Sensitivity ---
    "No, the test configuration uses ChromaDB, not PostgreSQL.",
    "No, the chunk size is about 500 tokens with a 50-token overlap.",
    "No, you should never commit directly to main.",
    "No, the maximum file size is 20 MB.",
]

from evaluation.rag.rag_evaluation import evaluate_rag
from src.rag.rag_pipeline import RagPipeline

pipeline = RagPipeline()

evaluate_rag(
    pipeline=pipeline,
    questions=questions,
    reference_answers=reference_answers,
)
