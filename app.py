
import streamlit as st
import requests

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="TeamDocs AI",
    page_icon="📚",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "connected" not in st.session_state:
    st.session_state.connected = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "repository" not in st.session_state:
    st.session_state.repository = None

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 TeamDocs AI")
st.caption("Ask questions about your GitHub repository using RAG and AI.")

# --------------------------------------------------
# Sidebar - Configuration
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    openrouter_api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="Enter your OpenRouter API key",
    )

    github_token = st.text_input(
        "GitHub Personal Access Token",
        type="password",
        placeholder="Enter your GitHub token",
    )

    st.divider()

    st.subheader("GitHub Repository")

    owner = st.text_input(
        "Repository Owner",
        value="HosseinHeydari2004",
    )

    repo = st.text_input(
        "Repository Name",
        value="TeamDocs-AI",
    )

    connect_button = st.button(
        "🔗 Connect Repository",
        use_container_width=True,
    )

    if connect_button:
        if not openrouter_api_key.strip():
            st.error("Please enter your OpenRouter API key.")

        elif not github_token.strip():
            st.error("Please enter your GitHub token.")

        elif not owner.strip() or not repo.strip():
            st.error("Please enter the repository owner and name.")

        else:
            st.session_state.connected = True
            st.session_state.repository = f"{owner}/{repo}"
            st.session_state.messages = []

            st.success("Repository configuration saved.")
            st.info("Your repository will be indexed when you send your first question.")

    if st.session_state.connected:
        st.divider()
        st.success(f"Connected to: {st.session_state.repository}")

# --------------------------------------------------
# Main Chat Interface
# --------------------------------------------------

if not st.session_state.connected:
    st.info("Enter your API credentials and repository details, then connect.")

else:
    st.subheader(f"💬 Chat with {st.session_state.repository}")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input("Ask a question about your repository...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing repository and generating answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "question": question,
                            "owner": owner,
                            "repo": repo,
                            "openrouter_api_key": openrouter_api_key,
                            "github_token": github_token,
                        },
                        timeout=300,
                    )

                    response.raise_for_status()
                    result = response.json()

                    answer = result.get(
                        "answer",
                        "No answer was returned.",
                    )

                    st.markdown(answer)

                    if result.get("index_status"):
                        st.caption(
                            f"Index status: {result['index_status']} | "
                            f"Documents: {result.get('document_count', 0)}"
                        )

                    if result.get("warnings"):
                        for warning in result["warnings"]:
                            st.warning(str(warning))

                except requests.exceptions.Timeout:
                    answer = "The request timed out. Please try again."

                    st.error(answer)

                except requests.exceptions.ConnectionError:
                    answer = "Cannot connect to FastAPI. Make sure the backend is running."

                    st.error(answer)

                except requests.exceptions.HTTPError:
                    answer = "The backend returned an error. Please check the backend logs."

                    st.error(answer)

                except Exception:
                    answer = "An unexpected error occurred."

                    st.error(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
