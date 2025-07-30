#!/usr/bin/env python3
"""
Streamlit Web Interface for Kavak Travel Assistant
"""

import streamlit as st
from main import Config, TravelAssistantAgent

def main():
    st.set_page_config(
        page_title="Kavak Travel Assistant",
        page_icon="✈️",
        layout="wide"
    )
    
    st.title("✈️ Kavak Travel Assistant")
    st.markdown("Your intelligent travel planning companion")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "assistant" not in st.session_state:
        config = Config()
        st.session_state.assistant = TravelAssistantAgent(config)
    
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This travel assistant helps you:
        - Find flights with natural language queries
        - Get visa and travel information
        - Plan your international trips
        
        **Example queries:**
        - "Find me a round-trip to Tokyo in August with Star Alliance airlines only"
        - "What are the visa requirements for UAE citizens visiting Japan?"
        - "Tell me about refund policies for airline tickets"
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me about flights, visas, or travel information..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.assistant.process_message(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 