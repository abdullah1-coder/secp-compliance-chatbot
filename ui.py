# ui.py
import streamlit as st
import requests
import uuid

# Configure professional corporate layout parameters
st.set_page_config(
    page_title="SECP Compliance Assistant", 
    page_icon="⚖️", 
    layout="centered"
)

st.title("SECP Compliance Assistant")
st.caption("Production Decoupled Interface. Connection channel routed via localized API gateways.")

# Initialize global chat session persistence arrays
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Ensure that the session history tracking payload resets cleanly on initial browser handshakes
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I am your SECP compliance assistant. How can I assist you with corporate filings or name reservations today?"
        }
    ]

# Render active conversational history logs to interface container spaces
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Primary user interaction loop
if user_input := st.chat_input("Ask about name reservations, fees, or forms..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("*Verifying statutory documentation parameters...*")
        
        try:
            # Route transaction request to backend microservice listening network port
            backend_url = "https://secp-compliance-chatbot.onrender.com/chat"
            payload = {
                "input": user_input, 
                "session_id": st.session_state.session_id
            }
            
            api_call = requests.post(backend_url, json=payload, timeout=90)
            if api_call.status_code == 200:
                bot_response = api_call.json().get("answer", "Empty transaction data returned.")
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                response_placeholder.markdown(bot_response)
            else:
                response_placeholder.markdown(f"Error: API Gateway issued status code {api_call.status_code}.")
        except Exception as e:
            response_placeholder.markdown("Connection Error: Failed to establish a connection with the localized backend server.")