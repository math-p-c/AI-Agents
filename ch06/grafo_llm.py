# /src/ch06/grafo_llm.py
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
import operator

load_dotenv()

# === ESTADO ===
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# === MODELO ===
modelo = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
        temperature=0)

# === NÓS ===
def preparar(state: ChatState) -> dict:
    """Adiciona system message se necessário."""
    messages = state["messages"]

    # Se não há system message, adicionar
    if not messages or not isinstance(messages[0], SystemMessage):
        system = SystemMessage(content="Você é um assistente prestativo. Responda em português.")
        return {"messages": [system]}

    return {"messages": []}

def chamar_modelo(state: ChatState) -> dict:
    """Chama o LLM."""
    resposta = modelo.invoke(state["messages"])
    return {"messages": [resposta]}

# === GRAFO ===
grafo = StateGraph(ChatState)

# Nós
grafo.add_node("preparar", preparar)
grafo.add_node("chamar_modelo", chamar_modelo)

# Arestas
grafo.add_edge(START, "preparar")
grafo.add_edge("preparar", "chamar_modelo")
grafo.add_edge("chamar_modelo", END)

# Compilar
app = grafo.compile()

# === USAR ===
def chat(mensagem: str) -> str:
    resultado = app.invoke({
        "messages": [HumanMessage(content=mensagem)]
    })
    return resultado["messages"][-1].content

# Testar
if __name__ == "__main__":
    print(chat("Qual é a capital da França?"))
    print("\n---\n")
    print(chat("E do Brasil?"))