# rag_engine.py
import os
import re
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Telemetry tracking configurations
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "SECP ChatBot"

# Primary inference engine node
llm_node = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)

# Global variables to cache the pipeline states in memory after the first request
_cached_chain = None

def get_production_chain():
    """
    Lazily initializes the RAG pipeline components.
    This guarantees that network requests are only made after the server is fully running.
    """
    global _cached_chain
    if _cached_chain is not None:
        return _cached_chain

    # Document ingestion layer
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()

    # Token split allocations
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Cloud hosted embedding mapping using zero local RAM
    huggingface_api_key = os.environ.get("HF_TOKEN")
    bge_embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=huggingface_api_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Seed spatial index workspace
    vectorstore = Chroma.from_documents(documents=splits, embedding=bge_embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Sparse text search matching index
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    # High-speed memory-efficient hybrid network ensemble
    production_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.4, 0.6]
    )

    # Multi-turn interaction context refactoring chains
    contextualize_q_system_prompt = "Given a chat history and the latest user question, re-phrase it to be standalone."
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm_node, production_retriever, contextualize_q_prompt)

    # Core prompt blueprint specifications
    qa_system_prompt = "You are an expert SECP corporate compliance assistant. Answer using the context:\n\n{context}"
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm_node, qa_prompt)
    rag_production_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Memory storage configuration mappings
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    _cached_chain = RunnableWithMessageHistory(
        rag_production_chain, 
        get_session_history,
        input_messages_key="input", 
        history_messages_key="chat_history", 
        output_messages_key="answer"
    )
    return _cached_chain

def redact_pakistani_pii(text: str) -> str:
    return re.sub(r'\b\d{5}-\d{7}-\d{1}\b', '[CNIC_REDACTED]', text)

def output_safety_filter(response_text: str) -> str:
    fallback_response = "I cannot process that request as it violates safety guidelines."
    if any(k in response_text.lower() for k in ["system prompt", "compliance assistant", "provided context blocks"]):
        return fallback_response
    return redact_pakistani_pii(response_text)