# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag_engine import conversational_production_rag, redact_pakistani_pii, output_safety_filter

app = FastAPI(
    title="SECP Compliance Engine Server",
    description="Production-grade core middleware routing tier for corporate regulatory assistance.",
    version="1.0.0"
)

# Configure Cross-Origin Resource Sharing policy restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def handle_rag_chat(payload: dict):
    """
    Processes inbound compliance queries via standard dictionary data mapping structures.
    Applies explicit input sanitation and output filters before transaction finalization.
    """
    raw_query = payload.get("input")
    session_id = payload.get("session_id")
    
    if not raw_query or not session_id:
        raise HTTPException(status_code=400, detail="Missing required payload parameters: 'input' and 'session_id'.")
    
    # Process pre-retrieval validation checks
    sanitized_query = redact_pakistani_pii(raw_query)
    if any(k in sanitized_query.lower() for k in ["system prompt", "ignore"]):
        return {"session_id": session_id, "answer": "I cannot process that request.", "status": "BLOCKED"}
        
    try:
        # Invoke decoupled AI runtime pipeline with session metadata profiles
        config = {"configurable": {"session_id": session_id}}
        chain_output = conversational_production_rag.invoke({"input": sanitized_query}, config=config)
        
        # Apply structural output verification filter pass
        final_bot_delivery = output_safety_filter(chain_output["answer"])
        return {"session_id": session_id, "answer": final_bot_delivery, "status": "SUCCESS"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Map network listening port parameters dynamically from container environmental variables
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)