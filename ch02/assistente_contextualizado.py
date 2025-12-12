# assistente_contextualizado.py
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

class AssistenteContextualizado:
    def __init__(self, nome_usuario: str):
        self.modelo = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
        temperature=0)
        self.nome_usuario = nome_usuario
        self.historico = []

    def _get_system_prompt(self) -> str:
        agora = datetime.now()
        return f"""Você é um assistente pessoal chamado Jarvis. Se for a primeira mensagem da conversa, cumprimente o usuário e se apresente.

## Contexto
- Usuário: {self.nome_usuario}
- Data: {agora.strftime('%d/%m/%Y')}
- Hora: {agora.strftime('%H:%M')}

## Personalidade
- Seja cordial e use o nome do usuário ocasionalmente
- Responda em português brasileiro
- Seja conciso, mas completo
"""

    def conversar(self, mensagem: str) -> str:
        # System prompt atualizado a cada chamada (data/hora atual)
        system = SystemMessage(content=self._get_system_prompt())

        # Montar mensagens: system + histórico + nova mensagem
        mensagens = [system] + self.historico + [HumanMessage(content=mensagem)]

        # Obter resposta
        resposta = self.modelo.invoke(mensagens)

        # Atualizar histórico (sem o system, que é recriado)
        self.historico.append(HumanMessage(content=mensagem))
        self.historico.append(resposta)

        return resposta.content

def main():
    nome = input("Qual é o seu nome? ").strip() or "Usuário"
    assistente = AssistenteContextualizado(nome_usuario=nome)

    print(f"\nOlá, {nome}! Sou o Jarvis, seu assistente pessoal.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        entrada = input("Você:\n").strip()
        if entrada.lower() == 'sair':
            print(f"Até logo, {nome}!")
            break
        if entrada:
            resposta = assistente.conversar(entrada)
            print(f"Jarvis:\n{resposta}")

if __name__ == "__main__":
    main()