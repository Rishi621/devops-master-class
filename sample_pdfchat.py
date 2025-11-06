import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os
import httpx
import uuid

# ---------------------
# HTTP client & Tiktoken
# ---------------------
client = httpx.Client(verify=False)
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# ---------------------
# LLM and Embeddings
# ---------------------
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
# Streamlit UI
# ---------------------
st.set_page_config(page_title="RAG PDF Chat & Summarizer")
st.title("📄 RAG-powered PDF Chat & Summarizer")

# Upload PDF
upload_file = st.file_uploader("Upload a PDF", type="pdf")

if upload_file:
    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(upload_file.read())
        tmp_file_path = tmp_file.name

    # Extract text
    raw_text = extract_text(tmp_file_path)

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # Unique folder per PDF
    pdf_id = str(uuid.uuid4())
    persist_dir = f"./chroma_index_{pdf_id}"

    # Create vector DB
    with st.spinner("Indexing document..."):
        vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory=persist_dir)
        vectordb.persist()

    # Create retriever
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # ---------------------
    # Summarization
    # ---------------------
    st.subheader("📝 Summary")
    summary_prompt = "Please summarize this document based on the key topics:"
    result_dict = rag_chain.invoke(summary_prompt)
    st.write(result_dict['result'])

    # ---------------------
    # Chat with PDF
    # ---------------------
    st.subheader("💬 Chat with PDF")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        response_dict = rag_chain.invoke(user_question)
        answer = response_dict['result']

        # Save chat history
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("PDF Bot", answer))

    # Display chat history
    for speaker, text in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {text}")