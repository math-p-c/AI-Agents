# hello_llm.py
import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Criar instância do modelo
modelo = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
    temperature=0,  # 0 = determinístico, 1 = criativo
    max_output_tokens=1024,
    max_retries=3,
    timeout=30,
)

#Criando loop para criar interação entre usuario e modelo:
while True:
    mensagem = HumanMessage(content=input("Você: "))
    # Invocar o modelo
    resposta = modelo.invoke([mensagem])
    # Exibir a resposta
    print(f"Resposta do modelo1: {resposta.content}")
    if 'Tchau' in resposta.content:
        break