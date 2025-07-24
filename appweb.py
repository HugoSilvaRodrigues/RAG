import streamlit as st
from IAagent import AgentState, agent_graph

st.title("ChatBot v1")

st.sidebar.title("Context")
st.sidebar.markdown("""
First version of the chatbot designed to answer questions about the use of artificial intelligence as a controller for nonlinear systems.
This is a highly complex topic with several nuances. To ensure scientifically grounded answers, this application includes a RAG database containing excerpts from my master's thesis. 
""")

st.sidebar.subheader("Current Limitations")
st.sidebar.markdown("""
- Answers are limited to:
  - The content in the RAG database
  - The AI model’s internal knowledge
""")

st.sidebar.subheader("Next Steps")
st.sidebar.markdown("""
                 - Enable web search capabilities to expand the model's access to external, up-to-date information
                 """)

contact=st.sidebar.button("Contact")
if contact:
    st.sidebar.write("hugosilvarodrigues@gmail.com")



tab1,tab2=st.tabs(["About the project","ChatBot"])
with tab2:
    st.title("ChatBot")
    query=st.text_input("Type your question: ")

    submit=st.button("Submit")
    st.spinner("")
    if submit:
        with st.spinner("In progress..."):
            responses= agent_graph.invoke(
            AgentState(query=query),
            output_keys=["possible_responses"]
            )
    
        st.subheader("LLM Answer")
        try:
            st.write(responses["possible_responses"][0]["answer"])
        except Exception as e:
            st.write(responses)
        
        st.subheader("RAG files")       
        
        try:
            st.write(responses["possible_responses"][0]["context"])
        except Exception as e:
            st.write("No files founded")

with tab1:
    st.title("Project Overview")

    st.subheader("How to Use")
    st.markdown("""
    The LLM will access the RAG system when it detects **specific keywords**, such as:

    - `based`
    - `explain`

    If these keywords are **not** used, the response will be generated **only by the model**, without retrieving documents from the database.

    ⚠️ More keywords and documents will be added soon to expand the RAG capabilities.
    """)

    st.subheader("LLM Information")
    st.markdown("""
    I'm using **LLaMA 3 8B** from Hugging Face:  
    [https://huggingface.co/lmstudio-community/Llama-3-Groq-8B-Tool-Use-GGUF](https://huggingface.co/lmstudio-community/Llama-3-Groq-8B-Tool-Use-GGUF)

    The model is being served via **LM Studio** to provide an API connection.
""")
