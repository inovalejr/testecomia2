"""
app.py — Agente RAG + DuckDB + LangGraph + Gradio focado na base 'Stoppers'
- Foco: respostas conversacionais (sem scores), e análises avançadas **apenas quando solicitadas**.
- Deploy-ready: demo.launch(server_name="0.0.0.0", server_port=PORT)
"""

from __future__ import annotations
import os
import sys
import time
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import duckdb
from pydantic import BaseModel
import gradio as gr

# Embeddings / RAG
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LangGraph
from langgraph.graph import StateGraph, END

# Optional OpenAI for LLM responses (if provided)
try:
    import openai
except Exception:
    openai = None

# -----------------------
# Config
# -----------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
STOPPERS_CSV = os.path.join(DATA_DIR, "stoppers.csv")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "embed_cache")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", 8080))
TOP_K = int(os.getenv("TOP_K", 20))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Logging
logging.basicConfig(stream=sys.stdout, level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stoppers_agent")

# -----------------------
# System Prompt (Master)
# -----------------------
SYSTEM_PROMPT = """
Você é um assistente especialista em Stoppers da Brisanet. Seu papel:
- Analisar registros de stoppers e responder de forma natural, consultiva e prática.
- Entender todas as colunas da base (códigos, nome do site, código stopper, tipo, status, descrição,
  criticidade, datas, solicitante, papel responsável, fornecedor, responsável, etc).
- Não mostre scores, IDs técnicos ou debug. Responda como um analista sênior.
- Realize validações técnicas (datas, coerência status/descriptions) e análises gerenciais apenas
  quando requisitado explicitamente (ex.: "qual % resolvido dentro do prazo no mês passado?").
- Sempre que apropriado, proponha ações concretas (quem acionar, próxima atividade).
"""

# -----------------------
# Utilities: load & normalize
# -----------------------
def load_stoppers(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"Arquivo não encontrado: {path}. Criando DataFrame vazio padrão.")
        cols = ["Códigos de sitio","Nombres de sitios","Nombre","Código","Código de workflow",
                "Código Stopper","Nome Stopper","Tipo de Stopper","Status","Descrição","Criticidade",
                "Data de solicitacao","Data vencimiento","Data de encerramento","hora de encerramento",
                "Atributo","solicitado por","Papel responsable","Fornecedor responsavel","Responsavel"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str).fillna("")
    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    return df

# -----------------------
# Semantic Index (SentenceTransformer) with cache
# -----------------------
class SemanticIndex:
    def __init__(self, df: pd.DataFrame, text_cols: List[str], model_name: str = EMBEDDING_MODEL):
        self.df = df.reset_index(drop=True)
        self.text_cols = text_cols
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        # Build corpus: join text columns into a single string per row
        self.corpus = (self.df[self.text_cols].fillna("").astype(str).agg(" \n ".join, axis=1)).tolist()
        self.emb = self._load_or_build()

    def cache_path(self) -> str:
        return os.path.join(EMBED_CACHE_DIR, "stoppers_emb.npz")

    def _load_or_build(self):
        path = self.cache_path()
        if os.path.exists(path):
            try:
                npz = np.load(path)
                emb = npz["emb"]
                if emb.shape[0] == len(self.corpus):
                    logger.info("Carregando embeddings de cache (stoppers).")
                    return emb
                else:
                    logger.info("Cache mismatch (tamanho). Recriando embeddings.")
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}")
        logger.info("Calculando embeddings (pode demorar na primeira execução)...")
        emb = self.model.encode(self.corpus, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        np.savez_compressed(path, emb=emb)
        logger.info(f"Embeddings salvos em cache: {path}")
        return emb

    def search(self, query: str, k:int = None) -> List[Tuple[int, float]]:
        if not query or str(query).strip()=="":
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.emb)[0]
        
        # Se k não especificado ou for muito grande, retornar todos os registros ordenados por relevância
        if k is None or k >= len(sims):
            topk = sims.argsort()[::-1]
        else:
            # Usar argpartition para encontrar os k maiores elementos mais eficientemente
            topk_indices = np.argpartition(sims, -k)[-k:]
            topk = topk_indices[np.argsort(sims[topk_indices])[::-1]]
        return [(int(i), float(sims[i])) for i in topk]

# -----------------------
# Advanced duckdb queries (only run when asked)
# -----------------------
def detect_advanced_request(text: str) -> bool:
    patterns = [r"\bpercent", r"\bpercentual", r"\bporcent", r"\bresolvid", r"dentro do prazo", r"taxa", r"mês", r"último", r"última", r"last month", r"\b%"]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def duckdb_percent_resolved_within(con: duckdb.DuckDBPyConnection, df_name: str, period: Optional[dict] = None) -> str:
    where = ""
    if period and period.get("year"):
        y = int(period["year"])
        m = int(period.get("month", 0))
        if m:
            where = f"WHERE strftime('%Y', Data_de_solicitacao)='{y}' AND strftime('%m', Data_de_solicitacao)='{m:02d}'"
        else:
            where = f"WHERE strftime('%Y', Data_de_solicitacao)='{y}'"

    query = f"""
    SELECT
        SUM(CASE WHEN Data_de_encerramento!='' AND Data_de_vencimiento!='' AND date(Data_de_encerramento) <= date(Data_de_vencimiento) THEN 1 ELSE 0 END) as dentro,
        SUM(CASE WHEN Data_de_encerramento!='' THEN 1 ELSE 0 END) as encerrados,
        COUNT(*) as total
    FROM {df_name}
    {where}
    """
    try:
        res = con.execute(query).fetchdf()
        if res.empty:
            return "Não foi possível calcular - sem registros."
        d = int(res.at[0,'dentro'] or 0)
        e = int(res.at[0,'encerrados'] or 0)
        tot = int(res.at[0,'total'] or 0)
        if tot == 0:
            return "Não há registros no período solicitado."
        pct = (d / tot) * 100
        return f"{pct:.1f}% dos {tot} stoppers foram resolvidos dentro do prazo (encerrados: {e}, dentro do prazo: {d})."
    except Exception as ex:
        logger.error(f"Erro em duckdb_percent_resolved_within: {ex}")
        return "Não foi possível calcular o percentual devido a formato de datas ou colunas."

def duckdb_group_count_percent(con: duckdb.DuckDBPyConnection, group_col: str, df_name: str, top_n: int = 6) -> str:
    try:
        q = f"""
        SELECT "{group_col}" as grupo, COUNT(*) as cnt
        FROM {df_name}
        GROUP BY "{group_col}"
        ORDER BY cnt DESC
        LIMIT {top_n}
        """
        df = con.execute(q).fetchdf()
        total = con.execute(f"SELECT COUNT(*) as total FROM {df_name}").fetchdf().at[0,'total'] or 0
        if total == 0:
            return "Sem registros para agrupar."
        lines = []
        for _, row in df.iterrows():
            grupo = row['grupo'] if row['grupo'] is not None else "SEM VALOR"
            cnt = int(row['cnt'])
            pct = (cnt/total)*100
            lines.append(f"- {grupo}: {cnt} ({pct:.1f}%)")
        return "Top ocorrências:\n" + "\n".join(lines)
    except Exception as ex:
        logger.error(f"Erro em duckdb_group_count_percent: {ex}")
        return "Erro ao agrupar por coluna solicitada. Verifique o nome da coluna."

# -----------------------
# LLMProvider: OpenAI or Mock
# -----------------------
class LLMProvider:
    def __init__(self, openai_key: str = OPENAI_API_KEY, model_name: str = OPENAI_MODEL):
        self.model = model_name
        self.use_openai = False
        if openai_key and openai is not None:
            openai.api_key = openai_key
            self.use_openai = True
            logger.info("LLMProvider: usando OpenAI")
        else:
            if openai_key and openai is None:
                logger.warning("OPENAI_API_KEY set but 'openai' package not available; using mock.")
            else:
                logger.info("OPENAI_API_KEY não setado; usando mock responder.")
            self.use_openai = False

    def answer(self, prompt: str, max_tokens:int=400) -> str:
        if self.use_openai:
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role":"system","content":SYSTEM_PROMPT}, {"role":"user","content":prompt}],
                    temperature=0,
                    max_tokens=max_tokens
                )
                return resp['choices'][0]['message']['content'].strip()
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                # fallback to mock
        # Mock: concise, safe reply (use prompt excerpt)
        excerpt = prompt.strip()[:800]
        return ("[MOCK RESPONSE - OPENAI não configurado]\n"
                "Resumo do pedido: " + (excerpt[:400] + ("..." if len(excerpt)>400 else "")) + 
                "\n\nResposta simulada: verifique a configuração de OPENAI_API_KEY para usar LLM real.")

# -----------------------
# LangGraph pipeline nodes
# -----------------------
class AgentState(BaseModel):
    pergunta: str
    top_k: Optional[int] = 20
    result: Optional[str] = None
    retrieved: Optional[List[Dict[str,Any]]] = None
    action_suggested: Optional[str] = None

def build_graph(retriever: SemanticIndex, con: duckdb.DuckDBPyConnection, df_table_name: str, llm: LLMProvider, safe_df: pd.DataFrame):
    graph = StateGraph(AgentState)

    def triagem(state: AgentState) -> AgentState:
        q = (state.pergunta or "").lower()
        if any(k in q for k in ["abrir chamado","abram chamado","abrir ticket"]):
            return {"action_suggested": "ABRIR_CHAMADO"}
        if detect_advanced_request(q):
            return {"action_suggested": "ANALISE_AVANCADA"}
        return {"action_suggested": "AUTO_RESOLVER"}

    def auto_resolver(state: AgentState) -> AgentState:
        q = state.pergunta or ""
        top_k = state.top_k
        logger.info(f"Buscando registros relevantes para: '{q[:50]}...' (modo: {'limitado' if top_k else 'completo'})")
        
        # Buscar todos os registros e filtrar por relevância (score > 0.1)
        hits = retriever.search(q, k=None)  # Buscar todos
        relevant_hits = [(idx, score) for idx, score in hits if score > 0.1]
        
        # Se o usuário especificou um limite, respeitar
        if top_k and top_k < len(relevant_hits):
            relevant_hits = relevant_hits[:top_k]
            logger.info(f"Limitando a {top_k} registros conforme solicitado")
        
        rows = []
        for idx, score in relevant_hits:
            row = retriever.df.iloc[idx].to_dict()
            snippet = " | ".join([f"{c}: {row.get(c,'')}" for c in retriever.text_cols if str(row.get(c,'')).strip()!=''])
            rows.append({"snippet": snippet})
        
        logger.info(f"Encontrados {len(rows)} registros relevantes (scores: {[f'{h[1]:.3f}' for h in relevant_hits[:3]]}...)")
        context = "\n\n".join([r["snippet"] for r in rows]) if rows else ""
        prompt = f"Contexto (trechos relevantes - {len(rows)} registros):\n{context}\n\nPergunta: {q}\n\nResponda de forma consultiva, prática e sucinta. Se o contexto for insuficiente, peça mais informação."
        answer = llm.answer(prompt)
        return {"result": answer, "retrieved": rows}

    def analise_avancada(state: AgentState) -> AgentState:
        q = (state.pergunta or "").lower()
        
        if "resolvid" in q and "prazo" in q or "dentro do prazo" in q or "%" in q or "percent" in q or "percentual" in q:
            period = None
            if "mês passado" in q or "mes passado" in q or "último mês" in q or "ultimo mes" in q:
                today = pd.Timestamp.now()
                last = today - pd.DateOffset(months=1)
                period = {"year": last.year, "month": last.month}
            
            try:
                con.execute(f"""
                CREATE OR REPLACE VIEW stoppers_work AS
                SELECT
                    "{get_col_safe(safe_df.columns, ['Data de solicitacao','Data_de_solicitacao','Data de solicitacao'])}" AS Data_de_solicitacao,
                    "{get_col_safe(safe_df.columns, ['Data de vencimiento','Data_de_vencimiento','Data de vencimiento'])}" AS Data_de_vencimiento,
                    "{get_col_safe(safe_df.columns, ['Data de encerramento','Data_de_encerramento','Data de encerramento'])}" AS Data_de_encerramento,
                    *
                FROM {df_table_name}
                """)
                res_text = duckdb_percent_resolved_within(con, "stoppers_work", period)
            except Exception as ex:
                logger.warning(f"Failed to create stoppers_work view, trying with original table. Error: {ex}")
                res_text = duckdb_percent_resolved_within(con, df_table_name, period)
            
            prompt = f"Usuário perguntou: {state.pergunta}\nResultado analítico bruto:\n{res_text}\n\nFormule uma resposta humana e consultiva com recomendações práticas."
            answer = llm.answer(prompt)
            return {"result": answer, "retrieved": None}
        
        match = re.search(r"por (\w+)", q)
        if match:
            col = match.group(1)
            mapping = {
                "criticidade": ["Criticidade","criticidade"],
                "tipo": ["Tipo de Stopper","Tipo","Tipo_de_Stopper"],
                "responsavel": ["Responsavel","responsavel","Papel responsable","Fornecedor responsavel"],
                "fornecedor": ["Fornecedor responsavel","fornecedor","Fornecedor"]
            }
            cand_cols = mapping.get(col, [col])
            chosen = get_col_safe(safe_df.columns, cand_cols, default=col)
            res_text = duckdb_group_count_percent(con, chosen, df_table_name, top_n=6)
            prompt = f"Usuário perguntou: {state.pergunta}\nResultado analítico bruto:\n{res_text}\n\nResponda de forma humana e sugira ações práticas."
            answer = llm.answer(prompt)
            return {"result": answer, "retrieved": None}

        # Fallback generic
        chosen_col = get_col_safe(safe_df.columns, ['Nome Stopper', 'Nome_Stopper'], default='Nome_Stopper')
        res_text = duckdb_group_count_percent(con, chosen_col, df_table_name, top_n=6)
        prompt = f"Usuário perguntou: {state.pergunta}\nResultado analítico bruto:\n{res_text}\n\nTransforme em resposta humana e consultiva."
        answer = llm.answer(prompt)
        return {"result": answer, "retrieved": None}

    def pedir_info(state: AgentState) -> AgentState:
        return {"result": "Preciso de mais detalhes para responder; por favor especifique o período, cidade ou coluna desejada.", "retrieved": None}

    def abrir_chamado(state: AgentState) -> AgentState:
        q = state.pergunta or ""
        template = (f"Solicitação de abertura de chamado gerada automaticamente.\nResumo: {q[:250]}\n"
                    "Prioridade sugerida: Alta. Próximo passo: notificar responsável do workflow.")
        return {"result": template, "retrieved": None}

    # Register nodes
    graph.add_node("triagem", triagem)
    graph.add_node("auto_resolver", auto_resolver)
    graph.add_node("analise_avancada", analise_avancada)
    graph.add_node("pedir_info", pedir_info)
    graph.add_node("abrir_chamado", abrir_chamado)

    # Define conditional edges based on action_suggested
    def route_after_triagem(state: AgentState) -> str:
        action = state.action_suggested
        if action == "ABRIR_CHAMADO":
            return "abrir_chamado"
        elif action == "ANALISE_AVANCADA":
            return "analise_avancada"
        elif action == "AUTO_RESOLVER":
            return "auto_resolver"
        else:
            return "pedir_info"

    # Set up the flow
    graph.set_entry_point("triagem")
    graph.add_conditional_edges("triagem", route_after_triagem, {
        "abrir_chamado": "abrir_chamado",
        "analise_avancada": "analise_avancada", 
        "auto_resolver": "auto_resolver",
        "pedir_info": "pedir_info"
    })
    
    # All end nodes go to END
    graph.add_edge("auto_resolver", END)
    graph.add_edge("analise_avancada", END)
    graph.add_edge("pedir_info", END)
    graph.add_edge("abrir_chamado", END)

    return graph.compile()

# -----------------------
# Helper functions for duckdb column mapping
# -----------------------
def retrieve_original_cols(con: duckdb.DuckDBPyConnection, table_name: str) -> List[str]:
    try:
        df = con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
        return [r['name'] for _, r in df.iterrows()]
    except Exception:
        return []

def column_exists(con: duckdb.DuckDBPyConnection, table_name: str, col_name: str) -> bool:
    cols = retrieve_original_cols(con, table_name)
    return col_name in cols

def get_col_safe(cols: List[str], candidates: List[str], default: Optional[str] = None) -> str:
    for c in candidates:
        if c in cols:
            return c
    return default or (candidates[0] if candidates else "")

def view_exists(con: duckdb.DuckDBPyConnection, view_name: str) -> bool:
    try:
        res = con.execute(f"SELECT name FROM sqlite_master WHERE type='view' AND name='{view_name}'").fetchdf()
        return not res.empty
    except Exception:
        return False

# -----------------------
# Gradio / main
# -----------------------
def main():
    # Load stoppers
    df = load_stoppers(STOPPERS_CSV)
    con = duckdb.connect(database=':memory:')
    safe_df = df.copy()
    
    # Normalize column names
    col_map = {}
    for col in safe_df.columns:
        safe_name = col.replace(" ", "_").replace("-", "_").replace(".", "").replace("/","_")
        if safe_name != col:
            safe_df = safe_df.rename(columns={col: safe_name})
            col_map[safe_name] = col
    
    # Register to duckdb
    con.register("stoppers", safe_df)
    table_name = "stoppers"

    # Build semantic index - usar TODAS as colunas da base de dados
    all_columns = safe_df.columns.tolist()
    logger.info(f"Usando TODAS as {len(all_columns)} colunas para embeddings: {all_columns}")
    sem_index = SemanticIndex(safe_df, all_columns)

    # LLM provider
    llm = LLMProvider()

    # Build LangGraph
    graph = build_graph(sem_index, con, table_name, llm, safe_df)

    # Conversation memory
    sessions: Dict[str, List[Dict[str,str]]] = {}

    # Gradio UI
    with gr.Blocks(title="Agente Stoppers - Brisanet") as demo:
        gr.Markdown("# Agente Stoppers — Brisanet\nPergunte sobre stoppers. Peça análises avançadas explicitamente (ex.: '% resolvidos dentro do prazo no mês passado').")
        user_input = gr.Textbox(lines=2, label="Sua pergunta")
        analysis_mode = gr.Radio(
            choices=["Análise Completa (todos os registros relevantes)", "Análise Limitada (máximo de registros)"],
            value="Análise Completa (todos os registros relevantes)",
            label="Modo de Análise"
        )
        top_k_slider = gr.Slider(minimum=5, maximum=500, value=50, step=5, label="Máximo de registros (apenas no modo limitado)", visible=False)
        session_id = gr.Textbox(value="default", label="Session ID (use para manter contexto)", visible=False)
        chat_output = gr.Textbox(lines=12, label="Resposta")
        btn = gr.Button("Enviar")

        def handle_question(question: str, mode: str, top_k: int, sess: str):
            sess = sess or "default"
            sessions.setdefault(sess, [])
            
            # Determinar se deve usar limite baseado no modo
            use_limit = mode == "Análise Limitada (máximo de registros)"
            limit_value = top_k if use_limit else None
            
            state_in = {"pergunta": question, "top_k": limit_value}
            out = graph.invoke(state_in)
            result = out.get("result") or "Sem resposta gerada."
            sessions[sess].append({"q": question, "a": result})
            return result

        def toggle_slider_visibility(mode: str):
            return gr.update(visible=mode == "Análise Limitada (máximo de registros)")

        analysis_mode.change(fn=toggle_slider_visibility, inputs=[analysis_mode], outputs=[top_k_slider])
        btn.click(fn=handle_question, inputs=[user_input, analysis_mode, top_k_slider, session_id], outputs=[chat_output])

    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)

if __name__ == "__main__":
    main()