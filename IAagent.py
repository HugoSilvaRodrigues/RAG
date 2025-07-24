import yaml 
from langchain_openai import ChatOpenAI  #Library to create connection with the model api, using the ChatOpenAI allows change the model from differents companies easily
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate #Giving the context for the model 
from langchain_core.runnables import RunnablePassthrough #Recieve a dictionary and pass it forward
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain #Retrieval chain is a process in RAG when we connect the query with relevant information that helps the model with the answer
                                                        #It`s necessary to pass a document retriver object that will retrive a list of documents and a runnable(prompt) that produces a output with a input  
from pydantic import BaseModel 
from langgraph.graph import StateGraph # Creation of a graph of decisions/states for helping the LLM to generate an answer, this states can be defined based on keywords 
                                       #for example if the input has words like google or search the internet the model will be forced to look for info in the internet. 
                                       # The library StateGraph emulate the process of decision of an LLM based on what the developers wants the model to do based on code
                                       #https://blog.langchain.com/langgraph/
from langchain_huggingface import HuggingFaceEmbeddings


with open("config.yaml") as file:
    config=yaml.safe_load(file)

llm=ChatOpenAI(name="llama-3-groq-8b-tool-use",
               openai_api_base="http://127.0.0.1:1234/v1",
               openai_api_key="lm-studio"
               )

embedding_model=HuggingFaceEmbeddings(model_name=config["modelname"])

vectordb=Chroma(persist_directory=config["dvectordb"],embedding_function=embedding_model)#Loading the vectordb
retriver=vectordb.as_retriever()#https://python.langchain.com/docs/how_to/vectorstore_retriever/

prompt=PromptTemplate.from_template ("You are an expert in AI, asnwer the question {input} based on {context}")
prompt_without_rag=PromptTemplate.from_template(("You are an expert in AI, asnwer the question {input}"))


chain= RunnablePassthrough() | prompt | llm | StrOutputParser()
    #1-step: (RunnablePassthrough() : Recieve a dictionary formatted as Ex {"input": input, "context":document} and pass it forward
    #2-step: Prompt defined previous that demand 2 variables coming from step 1
    #3-step: llm from llm studio, huggingface, open ai or Ollama that will recieve the prompt with the input and context
    #Format the response
    
chain_without_rag= RunnablePassthrough() | prompt_without_rag | llm | StrOutputParser()
    #1-step: (RunnablePassthrough() : Recieve a dictionary formatted as Ex {"input": input} and pass it forward
    #2-step: Prompt defined previous that demand 1 variable coming from step 1
    #3-step: llm from llm studio, huggingface, open ai or Ollama that will recieve the prompt with the input and context
    #Format the response

retrieval_chain=create_retrieval_chain(retriver,chain) # Creating a retrieval chain that will access the Vectordb and pass the context with the input for the model
# retriver (input) | RunnablePassthrough("input":input, "context":retriver output) | prompt(RunnablePassthrought output)| llm (prompt) | StrOutputParser(prompt output)
#Who can I use it? qa_chain.invoke({"input":"query"}) https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html

class AgentState(BaseModel): #Inherits the class BaseModel so the structure of the class must be like this
    query: str
    next_step: str = ""
    retrived_info:list=[]
    possible_responses: list=[]
    
    
def agent_decision_nextStep (state:AgentState):
    query=state.query.lower()
    if any(word in query for word in ["explain", "based on"]):
        state.next_step="retrieve"
    else:
        state.next_step="not retrieve"
    return state


def access_rag(state:AgentState)->AgentState:
    docs=retriver.invoke(state.query)#Using the retriver to obtain relevant documents based on similarity with the query
    state.retrived_info=docs
    return state

def generate_answers (state:AgentState)-> AgentState:
    responses=[retrieval_chain.invoke({"input": state.query}) for _ in range(1)]
    state.possible_responses=responses
    return state

def generate_answers_without_rag(state:AgentState)-> AgentState:
    responses=chain_without_rag.invoke({"input":state.query})
    state.possible_responses=responses
    return state

graph=StateGraph(AgentState)

#The LLM has 4 possible states:
graph.add_node("agent_decision_nextStep",agent_decision_nextStep)
graph.add_node("access_rag",access_rag)
graph.add_node("generate_answers",generate_answers)
graph.add_node("generate_answers_without_rag",generate_answers_without_rag)


#The graph start
graph.set_entry_point("agent_decision_nextStep")

#The llm can follow two paths: 
#Access the rag or not access the rag
graph.add_conditional_edges("agent_decision_nextStep",
                            lambda state: {
                                "retrieve" : "access_rag",
                                "not retrieve" : "generate_answers_without_rag"
                            } [state.next_step])

#Paths after the decision:

graph.add_edge("access_rag","generate_answers")

agent_graph=graph.compile()