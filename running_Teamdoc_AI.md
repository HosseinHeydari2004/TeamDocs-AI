## Running TeamDocs-AI

### Prerequisites
- Python 3.11
- Docker Desktop (installed and running)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/HosseinHeydari2004/TeamDocs-AI.git
cd TeamDocs-AI
```

### 2. Create the Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Start Docker

Open Docker Desktop and wait until the Docker Engine is running.

Verify the installation:

```powershell
docker info
```

The GitHub MCP server is started automatically through Docker when the application connects to GitHub. No separate MCP server installation is required.

### 4. Start the backend

From the project root:

```powershell
uvicorn devmind.api.main:app --reload
```

Backend API: http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

### 5. Start the frontend

Open a second terminal, activate the virtual environment, and run:

```powershell
streamlit run app.py
```

Open the Streamlit URL shown in the terminal (usually http://localhost:8501).

### 6. Connect a repository

In the Streamlit sidebar, enter:
- Your OpenRouter API key
- Your GitHub Personal Access Token
- Repository owner
- Repository name

Click **Connect Repository** and ask a question in the chat. Repository indexing starts automatically when you send the first question.

**Security:** Never commit API keys, GitHub tokens, or `.env` files to Git.