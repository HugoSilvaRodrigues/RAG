
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


model = ChatOpenAI(
    name="llama-3-groq-8b-tool-use",
    openai_api_base="http://127.0.0.1:1234/v1",
    openai_api_key="lm-studio"
)

# Define o estado do LangGraph
class State(TypedDict):
    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    documents: str
    next_step: str



def decide_next_step(state: State):
    state["next_step"] = "web"  
    return state

# Ferramenta manual
def search_internet(state: State):
    search = DuckDuckGoSearchRun()
    state["documents"] = search.invoke(state["query"])
    print(" DuckDuckGo Result:\n", state["documents"])
    return state

# Modelo responde usando documentos
def assistant(state: State):
    sys_message = SystemMessage(
        content="You are specialized in answer questions about AI as controllers for nonlinear systems. Use this context: "
        + state.get("documents", "")
    )
    print(sys_message)
    response = model.invoke([sys_message] + state["messages"])
    return {"messages": [response]}


# Construção do gráfico
builder = StateGraph(State)

builder.add_node("decide", decide_next_step)
builder.add_node("search_internet", search_internet)
builder.add_node("model", assistant)

builder.set_entry_point("decide")  

builder.add_conditional_edges(
    "decide",
    lambda state: {
        "web": "search_internet"
    }[state["next_step"]]
)

builder.add_edge("search_internet", "model")

react_agent = builder.compile()

# Execução do fluxo
state = {
    "query": "search on google how can i improve sliding mode control using ai",
    "messages": [],
    "documents": "",
    "next_step": ""
}

result = react_agent.invoke(state)

for m in result["messages"]:
    print(m.content)

    