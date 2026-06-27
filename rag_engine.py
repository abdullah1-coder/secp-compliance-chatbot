# rag_engine.py
import os
import re
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings  # Correct modern wrapper
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

# Telemetry tracking configurations
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "SECP ChatBot"

# Primary inference engine node
llm_node = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)

_cached_chain = None

def get_production_chain():
    """
    Lazily initializes the RAG pipeline components safely.
    Defers execution to avoid cold-boot network blocking or empty directory crashes.
    """
    global _cached_chain
    if _cached_chain is not None:
        return _cached_chain

    # Defensive path check: Create folder if it was ignored by Git
    target_dir = "Data"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Document ingestion layer
    loader = PyPDFDirectoryLoader(f"{target_dir}/")
    docs = loader.load()

    # Fallback structure: If folder has no PDFs, provide an explicit dummy document to prevent Chroma crash
    if not docs:
        from langchain_core.documents import Document
        splits = [Document(page_content="Securities and Exchange Commission of Pakistan (SECP) corporate compliance rules.")]
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

    # Cloud-hosted embedding mapping using zero local RAM via the updated partner class
    huggingface_api_key = os.environ.get("HF_TOKEN")
    
    # Explicit backup: Set the legacy variable string in memory so the internal LangChain client connects without crashing
    if huggingface_api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

    bge_embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=huggingface_api_key,
        task="feature-extraction"
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
    # Robust multi-turn interaction context refactoring chains
    contextualize_q_system_prompt = (
        "Given a chat history (which may contain markdown tables) and the latest user question, "
        "re-phrase the user question to be a standalone question that can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it. If you cannot reformulate it, return the user question exactly as it is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm_node, production_retriever, contextualize_q_prompt)

    # Core prompt blueprint specifications with explicit "not found" fallback handling
    qa_system_prompt = (
        "You are an expert SECP corporate compliance assistant. Answer the user's question using the provided context blocks. "
        "If the context does not contain the answer, or if you are unsure, explicitly state: "
        "'I am sorry, but the provided compliance documents do not contain specific information regarding this query.' "
        "Do not leave the response empty or blank.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm_node, qa_prompt)
    rag_production_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

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
    
    # Lowercase for uniform validation checks
    check_text = response_text.lower()
    
    # Target phrases that indicate the LLM leaked raw system configuration blueprints
    leak_indicators = [
        "you are an expert secp",
        "answer using the context",
        "provided context blocks",
        "qa_system_prompt"
    ]
    
    if any(phrase in check_text for phrase in leak_indicators):
        return fallback_response
        
    return redact_pakistani_pii(response_text)