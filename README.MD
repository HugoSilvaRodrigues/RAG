# Neural Network Control Signal Assistant

This project consists in using a **pre-trained LLM** model to answer questions about using **neural networks as control signals** to stabilize **uncertain non-linear systems**.

---

## 🧠 First Part — Vector Database

We build a **VectorDB** to assist the model in answering questions.  
The source files are provided in two formats: **JSON** and **PDF**, and contain sections from my **master's degree thesis**.

This VectorDB is implemented using the **LangChain** framework.

### 🔧 Important Libraries

- `langchain_chroma`: Vector database  
- `langchain_community.document_loaders`: For loading documents  
- `langchain.text_splitters`: To split document content  
- `langchain.huggingface`: For using a pre-trained embedding model from Hugging Face  

---

## 🤖 Second Part — AI Agent

Will be used a **pre-trained LLM** to act as the AI agent, with access to multiple sources of information such as:

- **RAG** (Retrieval-Augmented Generation)
- **Internet** (via APIs or plugins)

**Coming soon...**
