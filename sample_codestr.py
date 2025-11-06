import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import httpx
import tiktoken
import requests

# Patch requests to skip SSL verification (for tiktoken)
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# Set tiktoken cache directory
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# HTTP client
client = httpx.Client(verify=False)

# LLM and Embeddings setup
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

# Streamlit UI
st.set_page_config(page_title="RAG PDF Summarizer")
st.title("RAG-powered PDF Summarizer")

upload_file = st.file_uploader("Upload a PDF", type="pdf", key="pdf_upload")

if upload_file:
    # Step 1: Extract text
    raw_text = extract_text(upload_file)
    
    # Step 2: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    
    # Step 3: Embed and create vector DB
    with st.spinner("Indexing document..."):
        vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma_index")
        vectordb.persist()
    
    # Step 4: Create retriever
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Step 5: Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    # Step 6: Run summarization
    summary_prompt = "Please summarize this document based on the key topics:"
    result_dict = rag_chain.invoke(summary_prompt)
    
    # Extract answer and sources
    answer = result_dict['result']
    sources = result_dict['source_documents']
    
    # Display results
    st.subheader("📝 Summary")
    st.write(answer)
    
    st.subheader("📄 Source Documents")
    for doc in sources:
        st.write(doc.page_content)