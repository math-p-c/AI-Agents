# /src/chatbot.py
# Estudo de caso: Chatbot CRUD de Produtos com LangGraph
# Pratica conceitos dos capítulos 1, 2, 3, 5 e 6 do tutorial LangChain/LangGraph

#Dupla: Matheus Cardoso & Agda Silva

import os
import sqlite3
import operator
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

load_dotenv()

# === CONFIGURAÇÃO DO BANCO DE DADOS ===

DB_PATH = os.path.join(os.path.dirname(__file__), "produtos.db")


def inicializar_banco():
    """Cria a tabela de produtos se não existir."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS produtos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            preco REAL NOT NULL,
            estoque INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_conexao():
    """Retorna uma conexão com o banco de dados."""
    return sqlite3.connect(DB_PATH)


# === SCHEMAS PYDANTIC PARA VALIDAÇÃO (Cap 3) ===

class ProdutoInput(BaseModel):
    nome: str = Field(description="Nome do produto")
    preco: float = Field(description="Preço do produto em reais")
    estoque: int = Field(description="Quantidade em estoque")


class AtualizarProdutoInput(BaseModel):
    id: int = Field(description="ID do produto a ser atualizado")
    nome: Optional[str] = Field(default=None, description="Novo nome (opcional)")
    preco: Optional[float] = Field(default=None, description="Novo preço (opcional)")
    estoque: Optional[int] = Field(default=None, description="Novo estoque (opcional)")


# Input para atualizar por nome
class AtualizarPorNomeInput(BaseModel):
    nome: str = Field(description="Nome (ou parte) do produto a ser atualizado")
    novo_nome: Optional[str] = Field(default=None, description="Novo nome (opcional)")
    preco: Optional[float] = Field(default=None, description="Novo preço (opcional)")
    estoque: Optional[int] = Field(default=None, description="Novo estoque (opcional)")


# === TOOLS PARA CRUD (Cap 3) ===

@tool(args_schema=ProdutoInput)
def criar_produto(nome: str, preco: float, estoque: int) -> str:
    """Cria um novo produto no cadastro.

    Use esta ferramenta quando o usuário quiser cadastrar, adicionar ou
    criar um novo produto no sistema.
    """
    try:
        conn = get_conexao()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO produtos (nome, preco, estoque) VALUES (?, ?, ?)",
            (nome, preco, estoque)
        )
        conn.commit()
        produto_id = cursor.lastrowid
        conn.close()
        return f"Produto criado com sucesso! ID: {produto_id}, Nome: {nome}, Preço: R$ {preco:.2f}, Estoque: {estoque} unidades"
    except Exception as e:
        return f"Erro ao criar produto: {e}"


@tool
def listar_produtos(filtro_nome: Optional[str] = None) -> str:
    """Lista os produtos cadastrados.

    Use esta ferramenta quando o usuário quiser ver, listar, consultar ou
    buscar produtos. Pode filtrar por nome se especificado.

    Args:
        filtro_nome: Texto para filtrar produtos pelo nome (opcional)
    """
    try:
        conn = get_conexao()
        cursor = conn.cursor()

        if filtro_nome:
            cursor.execute(
                "SELECT id, nome, preco, estoque FROM produtos WHERE nome LIKE ?",
                (f"%{filtro_nome}%",)
            )
        else:
            cursor.execute("SELECT id, nome, preco, estoque FROM produtos")

        produtos = cursor.fetchall()
        conn.close()

        if not produtos:
            if filtro_nome:
                return f"Nenhum produto encontrado com '{filtro_nome}' no nome."
            return "Nenhum produto cadastrado."

        resultado = f"Encontrados {len(produtos)} produto(s):\n"
        for p in produtos:
            resultado += f"\n[{p[0]}] {p[1]} - R$ {p[2]:.2f} ({p[3]} em estoque)"

        return resultado
    except Exception as e:
        return f"Erro ao listar produtos: {e}"


@tool(args_schema=AtualizarProdutoInput)
def atualizar_produto(id: int, nome: Optional[str] = None, preco: Optional[float] = None, estoque: Optional[int] = None) -> str:
    """Atualiza os dados de um produto existente.

    Use esta ferramenta quando o usuário quiser modificar, alterar ou
    atualizar informações de um produto (nome, preço ou estoque).
    """
    try:
        conn = get_conexao()
        cursor = conn.cursor()

        # Verificar se produto existe
        cursor.execute("SELECT nome FROM produtos WHERE id = ?", (id,))
        produto = cursor.fetchone()
        if not produto:
            conn.close()
            return f"Produto com ID {id} não encontrado."

        # Construir query de atualização
        campos = []
        valores = []

        if nome is not None:
            campos.append("nome = ?")
            valores.append(nome)
        if preco is not None:
            campos.append("preco = ?")
            valores.append(preco)
        if estoque is not None:
            campos.append("estoque = ?")
            valores.append(estoque)

        if not campos:
            conn.close()
            return "Nenhum campo para atualizar foi informado."

        valores.append(id)
        query = f"UPDATE produtos SET {', '.join(campos)} WHERE id = ?"
        cursor.execute(query, valores)
        conn.commit()
        conn.close()

        atualizados = []
        if nome is not None:
            atualizados.append(f"nome='{nome}'")
        if preco is not None:
            atualizados.append(f"preço=R$ {preco:.2f}")
        if estoque is not None:
            atualizados.append(f"estoque={estoque}")

        return f"Produto ID {id} atualizado! Novos valores: {', '.join(atualizados)}"
    except Exception as e:
        return f"Erro ao atualizar produto: {e}"


@tool
def excluir_produto(id: int) -> str:
    """Exclui um produto do cadastro.

    Use esta ferramenta quando o usuário quiser remover, excluir ou
    deletar um produto do sistema.

    Args:
        id: ID do produto a ser excluído
    """
    try:
        conn = get_conexao()
        cursor = conn.cursor()

        # Verificar se produto existe
        cursor.execute("SELECT nome FROM produtos WHERE id = ?", (id,))
        produto = cursor.fetchone()
        if not produto:
            conn.close()
            return f"Produto com ID {id} não encontrado."

        nome_produto = produto[0]
        cursor.execute("DELETE FROM produtos WHERE id = ?", (id,))
        conn.commit()
        conn.close()

        return f"Produto '{nome_produto}' (ID {id}) excluído com sucesso!"
    except Exception as e:
        return f"Erro ao excluir produto: {e}"


# === CONFIGURAÇÃO DO AGENTE ===


@tool(args_schema=AtualizarPorNomeInput)
def atualizar_produto_por_nome(nome: str, novo_nome: Optional[str] = None, preco: Optional[float] = None, estoque: Optional[int] = None) -> str:
    """Atualiza um produto localizado pelo nome usando a tool `atualizar_produto`.

    - Se houver correspondência exata única, chama `atualizar_produto` por ID.
    - Se houver múltiplas correspondências, retorna a lista com IDs para o usuário escolher.
    - Se não houver correspondência, tenta busca parcial (LIKE).
    """
    try:
        conn = get_conexao()
        cursor = conn.cursor()

        # busca exata
        cursor.execute("SELECT id, nome FROM produtos WHERE nome = ?", (nome,))
        rows = cursor.fetchall()

        # se não encontrou, tentar busca parcial
        if not rows:
            cursor.execute("SELECT id, nome FROM produtos WHERE nome LIKE ?", (f"%{nome}%",))
            rows = cursor.fetchall()

        conn.close()

        if not rows:
            return f"Nenhum produto encontrado com nome '{nome}'."

        if len(rows) > 1:
            resp = "; ".join([f"[{r[0]}] {r[1]}" for r in rows])
            return "Múltiplos produtos encontrados: " + resp

        prod_id = rows[0][0]

        if novo_nome is None and preco is None and estoque is None:
            return "Nenhum campo para atualizar foi informado. Informe ao menos um campo (novo_nome, preco ou estoque)."

        args = {"id": prod_id}
        if novo_nome is not None:
            args["nome"] = novo_nome
        if preco is not None:
            args["preco"] = preco
        if estoque is not None:
            args["estoque"] = estoque

        # Reutilizar a tool existente para atualizar por ID
        try:
            return atualizar_produto.invoke(args)
        except Exception:
            # fallback chamando diretamente a função caso invoke não exista
            return atualizar_produto(**args)
    except Exception as e:
        return f"Erro ao atualizar por nome: {e}"


# Input para filtrar por estoque
class EstoqueFiltroInput(BaseModel):
    limite: int = Field(description="Retorna produtos com estoque menor que este valor")


@tool(args_schema=EstoqueFiltroInput)
def produtos_estoque_abaixo(limite: int) -> str:
    """Retorna produtos cujo `estoque` é menor que `limite`.

    Args:
        limite: inteiro; retorna produtos com estoque < limite
    """
    try:
        conn = get_conexao()
        cursor = conn.cursor()
        cursor.execute("SELECT id, nome, preco, estoque FROM produtos WHERE estoque < ? ORDER BY estoque ASC", (limite,))
        produtos = cursor.fetchall()
        conn.close()

        if not produtos:
            return f"Nenhum produto com estoque abaixo de {limite}."

        resultado = f"Produtos com estoque abaixo de {limite} ({len(produtos)}):\n"
        for p in produtos:
            resultado += f"\n[{p[0]}] {p[1]} - R$ {p[2]:.2f} ({p[3]} em estoque)"

        return resultado
    except Exception as e:
        return f"Erro ao listar produtos por estoque: {e}"


ALL_TOOLS = [criar_produto, listar_produtos, atualizar_produto, excluir_produto, atualizar_produto_por_nome, produtos_estoque_abaixo]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

SYSTEM_PROMPT = """Você é um assistente de gestão de estoque de produtos.

## Suas Capacidades
- Criar novos produtos (nome, preço, estoque)
- Listar produtos cadastrados (com filtro opcional por nome)
- Atualizar dados de produtos existentes
- Excluir produtos do cadastro
- Listar produtos com estoque abaixo de x

## Instruções
- Responda sempre em português brasileiro
- Seja amigável e prestativo
- Confirme as operações realizadas mostrando os detalhes
- Se o usuário não especificar todos os dados necessários, pergunte
- Para preços, use formato em reais (R$)
- Para estoque, use unidades inteiras

## Exemplos de uso
- "Cadastre um notebook por R$ 2500 com 10 unidades"
- "Liste todos os produtos"
- "Busque produtos com 'celular' no nome"
- "Atualize o preço do produto 1 para R$ 1999"
- "Exclua o produto 5"
"""


# === ESTADO DO AGENTE (Cap 6) ===

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# === CONFIGURAR MODELO ===

modelo = ChatAnthropic(
model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
temperature=0
)
modelo_com_tools = modelo.bind_tools(ALL_TOOLS)


# === NÓS DO GRAFO (Cap 6) ===

def no_llm(state: AgentState) -> dict:
    """Nó que chama o LLM com as tools bindadas."""
    messages = state["messages"]

    # Adicionar system prompt se não existir
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = modelo_com_tools.invoke(messages)
    return {"messages": [response]}


def no_tools(state: AgentState) -> dict:
    """Nó que executa as ferramentas chamadas pelo LLM."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []

    # Verificar se é AIMessage com tool_calls
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        try:
            result = TOOLS_BY_NAME[tool_call["name"]].invoke(tool_call["args"])
        except Exception as e:
            result = f"Erro ao executar {tool_call['name']}: {e}"

        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_messages}


def rotear(state: AgentState) -> Literal["tools", "__end__"]:
    """Decide se deve executar tools ou finalizar."""
    messages = state["messages"]
    last_message = messages[-1]

    # Verificar se é AIMessage com tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "__end__"


# === CONSTRUIR E COMPILAR O GRAFO (Cap 6) ===

def criar_agente():
    """Cria e retorna o agente compilado com checkpointer."""
    graph = StateGraph(AgentState)

    # Adicionar nós
    graph.add_node("llm", no_llm)
    graph.add_node("tools", no_tools)

    # Adicionar arestas
    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        rotear,
        {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "llm")

    # Compilar com checkpointer para persistência de sessão
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# === LOOP PRINCIPAL ===

def main():
    # Inicializar banco de dados
    inicializar_banco()

    # Criar agente
    agente = criar_agente()

    # Configuração da thread para persistência
    config: RunnableConfig = {"configurable": {"thread_id": "sessao-produtos"}}

    print("=" * 50)
    print("  CHATBOT DE GESTÃO DE PRODUTOS")
    print("=" * 50)
    print("Comandos: 'sair' para encerrar, 'limpar' para nova sessão")
    print("-" * 50)

    while True:
        try:
            entrada = input("\nVocê: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nEncerrando...")
            break

        if not entrada:
            continue

        if entrada.lower() == "sair":
            print("Até logo!")
            break

        if entrada.lower() == "limpar":
            # Criar nova sessão
            config: RunnableConfig = {"configurable": {"thread_id": f"sessao-{os.urandom(4).hex()}"}}
            print("Sessão limpa! Iniciando nova conversa.")
            continue

        # Invocar agente
        resultado = agente.invoke(
            {"messages": [HumanMessage(content=entrada)]},
            config=config
        )

        # Exibir resposta
        resposta = resultado["messages"][-1].content
        print(f"\nAssistente: {resposta}")


if __name__ == "__main__":
    main()

# TODO: permitir atualizar estoque pelo nome do produto, não só pelo ID
# TODO: retornar produtos com estoque abaixo de x
 