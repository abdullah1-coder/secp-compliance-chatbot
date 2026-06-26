# app.py (Updated with explicit exception debugging)
import os
import traceback  # Imported to catch the untruncated error graph
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

@app.post("/chat")
async def handle_rag_chat(payload: dict):
    raw_query = payload.get("input")
    session_id = payload.get("session_id")
    
    if not raw_query or not session_id:
        raise HTTPException(status_code=400, detail="Missing input or session_id payload parameters.")
    
    sanitized_query = redact_pakistani_pii(raw_query)
    if "system prompt" in sanitized_query.lower() or "ignore" in sanitized_query.lower():
        return {"session_id": session_id, "answer": "I cannot process that request.", "status": "BLOCKED"}
        
    try:
        conversational_production_rag = get_production_chain()
        config = {"configurable": {"session_id": session_id}}
        chain_output = conversational_production_rag.invoke({"input": sanitized_query}, config=config)
        final_bot_delivery = output_safety_filter(chain_output["answer"])
        return {"session_id": session_id, "answer": final_bot_delivery, "status": "SUCCESS"}
    except Exception as e:
        # Capture the entire un-truncated error traceback text string
        error_message = traceback.format_exc()
        # Return the raw error as the answer so it prints right on your Streamlit interface
        return {"session_id": session_id, "answer": f"⚠️ Backend Crash Details:\n\n{error_message}", "status": "ERROR"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)