# /src/ch03/assistente_com_tools.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

# === TOOLS ===

@tool
def celsiustof(c: float) -> str:
    """Converte Celsius para Fahrenheit."""
    return str(c * 1.8 + 32)

@tool
def celsiustokelvin(c: float) -> str:
    """Converte Celsius para Kelvin."""
    return str(c + 273.15)

@tool
def kelvintocelsius(k: float) -> str:
    """Converte Kelvin para Celsius."""
    return str(k - 273.15)

@tool
def ftocelsius(f: float) -> str:
    """Converte Fahrenheit para Celsius."""
    return str((f - 32) / 1.8)

@tool
def celsiustokandf(c: float) -> str:
    """Converte Celsius para Kelvin e para Fahrenheit."""
    responses = []
    responses.append(f"Celsius para Fahrenheit: {c * 1.8 + 32}")
    responses.append(f"Celsius para Kelvin: {c + 273.15}")
    return "\n".join(responses)


# === ASSISTENTE ===

class ConversorTemperaturas:
    def __init__(self):
        self.tools = [celsiustof, ftocelsius, celsiustokelvin]
        self.tools_por_nome = {t.name: t for t in self.tools}

        modelo = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
        temperature=0)
        self.modelo = modelo.bind_tools(self.tools)

        self.system = SystemMessage(content="""
Você é conversor de temperaturas.
Use as ferramentas disponíveis para fazer conversões entre Celsius, Kelvin e Fahrenheit.
""")

    def processar(self, pergunta: str) -> str:
        mensagens = [self.system, HumanMessage(content=pergunta)]

        while True:
            resposta = self.modelo.invoke(mensagens)
            mensagens.append(resposta)

            # Se não há tool_calls, retornar resposta final
            if not resposta.tool_calls:
                return resposta.content

            # Executar cada tool
            for tool_call in resposta.tool_calls:
                nome = tool_call["name"]
                args = tool_call["args"]
                call_id = tool_call["id"]

                resultado = self.tools_por_nome[nome].invoke(args)

                mensagens.append(ToolMessage(
                    content=resultado,
                    tool_call_id=call_id
                ))

def main():
    calc = ConversorTemperaturas()

    print("=== Conversor de Temperaturas ===")
    print("Digite 'sair' para encerrar.\n")

    while True:
        entrada = input("Você: ").strip()
        if entrada.lower() == 'sair':
            break
        if entrada:
            resposta = calc.processar(entrada)
            print(f"Conversor Temperaturas: {resposta}\n")

if __name__ == "__main__":
    main()