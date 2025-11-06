import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import uuid
import httpx
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ---------------------
# 🌐 Page Config
# ---------------------
st.set_page_config(page_title="🔍 Explainable RAG Chat", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.chat-bubble {
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 80%;
}
.user-bubble {
    background-color: #DCF8C6;
    margin-left: auto;
}
.bot-bubble {
    background-color: #E4E6EB;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# ---------------------
# 🧠 Models
# ---------------------
client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-Z6ZvcQAwUGMOReylW5me4Q",
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-Z6ZvcQAwUGMOReylW5me4Q",
    http_client=client
)

# ---------------------
# 🧩 Helper Functions
# ---------------------
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def extract_text_from_url(url):
    response = requests.get(url, verify=False, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "footer", "header", "nav"]):
        tag.decompose()
    elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
    return "\n".join(clean_text(el.get_text()) for el in elements if el.get_text())

def chunk_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    return [c for c in chunks if len(c.split()) > 5]

def create_vectordb_from_text(chunks):
    web_id = str(uuid.uuid4())
    persist_dir = f"./chroma_index_web_{web_id}"
    vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def summarize_text(text):
    summary_prompt = f"Summarize this webpage in under 100 words:\n{text[:4000]}"
    try:
        response = llm.invoke(summary_prompt)
        return getattr(response, "content", str(response))
    except Exception as e:
        return f"⚠️ Summary failed: {e}"

def explain_reasoning(question, answer, sources_text):
    explain_prompt = f"""
You are an explainable AI assistant.
The user asked: "{question}"
You answered: "{answer}"

The following text chunks were retrieved as evidence:
{sources_text}

Explain, in plain language, how these retrieved chunks support your answer.
List which chunks contributed most and why.
"""
    try:
        response = llm.invoke(explain_prompt)
        return getattr(response, "content", str(response))
    except Exception as e:
        return f"❌ Error generating explanation: {e}"

# ---------------------
# ⚙️ Session Management
# ---------------------
if "web_chains" not in st.session_state:
    st.session_state.web_chains = {}

# ---------------------
# 🧭 Sidebar
# ---------------------
st.sidebar.header("⚙️ Settings")
chunk_size = st.sidebar.slider("Chunk size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 50, 400, 200, 50)
retrieval_k = st.sidebar.slider("Top-K results", 2, 10, 5, 1)
if st.sidebar.button("🧹 Clear All Indexed Data"):
    st.session_state.web_chains.clear()
    st.sidebar.success("All data cleared!")

# ---------------------
# 🌐 URL Input & Indexing
# ---------------------
st.title("🔍 Explainable RAG Chat")
url = st.text_input("Enter a webpage URL:")

if url and url not in st.session_state.web_chains:
    try:
        with st.spinner(f"Fetching and indexing {url}..."):
            start_time = time.time()
            raw_text = extract_text_from_url(url)
            chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
            vectordb = create_vectordb_from_text(chunks)
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": retrieval_k})
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm, retriever=retriever, return_source_documents=True
            )
            summary = summarize_text(raw_text)
            st.session_state.web_chains[url] = {
                "vectordb": vectordb,
                "rag_chain": rag_chain,
                "chat_history": [],
                "summary": summary
            }

        st.success(f"✅ Indexed in {time.time() - start_time:.2f}s")

        with st.expander("📘 Webpage Summary", expanded=False):
            st.write(summary)

    except Exception as e:
        st.error(f"❌ Failed to index webpage: {e}")

# ---------------------
# 💬 Chat Interface
# ---------------------
if st.session_state.web_chains:
    selected_url = st.selectbox("Select webpage to chat with:", list(st.session_state.web_chains.keys()))
    web_data = st.session_state.web_chains[selected_url]

    st.subheader("💬 Ask Questions About the Webpage")
    user_question = st.text_input("Enter your question:")

    # 🚀 Handle Send Button
    if st.button("🚀 Send") and user_question:
        start = time.time()
        response_dict = web_data["rag_chain"].invoke(user_question)
        answer = response_dict["result"]
        source_docs = response_dict["source_documents"]

        web_data["chat_history"].append(("user", user_question))
        web_data["chat_history"].append(("bot", answer))

        # Store latest QA context in session
        st.session_state["last_answer"] = answer
        st.session_state["last_sources"] = source_docs
        st.session_state["last_question"] = user_question

        st.markdown(f"**Response time:** {time.time() - start:.2f}s")

        with st.expander("🔍 Show Retrieved Sources", expanded=False):
            for i, doc in enumerate(source_docs[:retrieval_k]):
                st.markdown(f"**Source Chunk #{i+1}:**")
                st.write(doc.page_content[:600] + "...")
                st.markdown("---")

        st.success("✅ Response generated! Now you can click '🧠 Explain my answer' below.")

    # 🧩 Explainability Reasoning (persistent after button click)
    if st.checkbox("🧠 Explain my answer") and "last_answer" in st.session_state:
        st.write("⚙️ Generating explanation...")
        with st.spinner("Generating explanation..."):
            try:
                sources_text = "\n\n".join(d.page_content[:500] for d in st.session_state["last_sources"])
                explanation = explain_reasoning(
                    st.session_state["last_question"],
                    st.session_state["last_answer"],
                    sources_text
                )
                st.write("✅ Explanation received!")
                with st.expander("🧩 Model’s Explanation", expanded=True):
                    st.write(explanation)
            except Exception as e:
                st.error(f"❌ Error generating explanation: {e}")

    # 🧼 Clear chat
    if st.button("Clear Chat History"):
        web_data["chat_history"] = []

    # 🗂️ Collapsible Chat History
    with st.expander("🕓 Chat History", expanded=False):
        for speaker, text in web_data["chat_history"]:
            bubble_class = "user-bubble" if speaker == "user" else "bot-bubble"
            st.markdown(f'<div class="chat-bubble {bubble_class}">{text}</div>', unsafe_allow_html=True)