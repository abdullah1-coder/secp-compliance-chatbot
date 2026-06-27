# app.py
import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag_engine import get_production_chain, redact_pakistani_pii, output_safety_filter

app = FastAPI(title="SECP Compliance Engine Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global initialization on startup to avoid request timeout errors
print("Booting Server: Warming up the SECP RAG Hybrid Engine...")
try:
    GLOBAL_RAG_CHAIN = get_production_chain()
    print("Success: All compliance documents loaded and indexed into Chroma successfully!")
except Exception as e:
    print("Critical Error during vector database startup compilation:")
    print(traceback.format_exc())
    GLOBAL_RAG_CHAIN = None

@app.post("/chat")
async def handle_rag_chat(payload: dict):
    global GLOBAL_RAG_CHAIN
    raw_query = payload.get("input")
    session_id = payload.get("session_id")
    
    if not raw_query or not session_id:
        raise HTTPException(status_code=400, detail="Missing input or session_id payload parameters.")
    
    if GLOBAL_RAG_CHAIN == None:
        try:
            GLOBAL_RAG_CHAIN = get_production_chain()
        except Exception as e:
            return {"session_id": session_id, "answer": f"Database Boot Error:\n\n{traceback.format_exc()}", "status": "ERROR"}

    sanitized_query = redact_pakistani_pii(raw_query)
    if "system prompt" in sanitized_query.lower() or "ignore" in sanitized_query.lower():
        return {"session_id": session_id, "answer": "I cannot process that request.", "status": "BLOCKED"}
        
    try:
        config = {"configurable": {"session_id": session_id}}
        chain_output = GLOBAL_RAG_CHAIN.invoke({"input": sanitized_query}, config=config)
        final_bot_delivery = output_safety_filter(chain_output["answer"])
        return {"session_id": session_id, "answer": final_bot_delivery, "status": "SUCCESS"}
    except Exception as e:
        error_message = traceback.format_exc()
        return {"session_id": session_id, "answer": f"Backend Crash Details:\n\n{error_message}", "status": "ERROR"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)