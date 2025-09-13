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
TOP_K = int(os.getenv("TOP_K", 6))
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
    # Standardize known Portuguese column names variants
    # (no rename forced, we'll access by names as-is, but ensure common ones exist)
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

    def search(self, query: str, k:int = TOP_K) -> List[Tuple[int, float]]:
        if not query or str(query).strip()=="":
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.emb)[0]
        topk = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in topk]

# -----------------------
# Advanced duckdb queries (only run when asked)
# -----------------------
def detect_advanced_request(text: str) -> bool:
    # triggers for advanced analytics: percent, resolved within, taxa, percentual, "mês", last month, "resolvidos dentro"
    patterns = [r"\bpercent", r"\bpercentual", r"\bporcent", r"\bresolvid", r"dentro do prazo", r"taxa", r"mês", r"último", r"última", r"last month", r"\b%"]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def duckdb_percent_resolved_within(con: duckdb.DuckDBPyConnection, df_name: str, period: Optional[dict] = None) -> str:
    """
    Exemplo de análise: % de stoppers resolvidos dentro do prazo.
    period: optional dict like {"year":2024,"month":8} to limit by month of solicitation or vencimento.
    """
    where = ""
    if period and period.get("year"):
        # assume Data de solicitacao in YYYY-MM-DD or similar
        y = int(period["year"])
        m = int(period.get("month", 0))
        if m:
            where = f"WHERE strftime('%Y', Data_de_solicitacao)='{y}' AND strftime('%m', Data_de_solicitacao)='{m:02d}'"
        else:
            where = f"WHERE strftime('%Y', Data_de_solicitacao)='{y}'"

    # We need to ensure column names are safe; map user columns to normalized ones in df_name
    query = f"""
    SELECT
        SUM(CASE WHEN Data_de_encerramento!='' AND Data_de_vencimiento!='' AND date(Data_de_encerramento) <= date(Data_de_vencimiento) THEN 1 ELSE 0 END) as dentro,
        SUM(CASE WHEN Data_de_encerramento!='' THEN 1 ELSE 0 END) as encerrados,
        COUNT(*) as total
    FROM {df_name}
    {where}
    """
    # Try multiple possible column name spellings - we'll adapt query to actual column names in the temp table
    # But for simplicity below, assume we created a temp view with aliased columns: Data_de_solicitacao, Data_de_vencimiento, Data_de_encerramento
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
    """
    Agrupa por uma coluna e retorna top N com percentuais
    """
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
    result: Optional[str] = None
    retrieved: Optional[List[Dict[str,Any]]] = None
    action_suggested: Optional[str] = None

def build_graph(retriever: SemanticIndex, con: duckdb.DuckDBPyConnection, df_table_name: str, llm: LLMProvider):
    graph = StateGraph(AgentState)

    def triagem(state: AgentState) -> AgentState:
        q = (state.pergunta or "").lower()
        # Simple keyword triage to decide path
        if any(k in q for k in ["abrir chamado","abram chamado","abrir ticket","abrir chamado"]):
            # direct open ticket
            return {"action_suggested": "ABRIR_CHAMADO"}
        # If user asks for statistics/percentual -> advanced analytics
        if detect_advanced_request(q):
            return {"action_suggested": "ANALISE_AVANCADA"}
        # If user asks simple verification or asks about a specific stopper -> try auto-resolver (retrieve context + answer)
        return {"action_suggested": "AUTO_RESOLVER"}

    def auto_resolver(state: AgentState) -> AgentState:
        q = state.pergunta or ""
        # semantic retrieve top K
        hits = retriever.search(q, k=TOP_K)
        rows = []
        for idx, score in hits:
            row = retriever.df.iloc[idx].to_dict()
            # build snippet that is human readable (no scores shown to user)
            snippet = " | ".join([f"{c}: {row.get(c,'')}" for c in retriever.text_cols if str(row.get(c,'')).strip()!=''])
            rows.append({"snippet": snippet})
        # build prompt for LLM: include system prompt implicitly via LLMProvider, but give context
        context = "\n\n".join([r["snippet"] for r in rows]) if rows else ""
        prompt = f"Contexto (trechos relevantes):\n{context}\n\nPergunta: {q}\n\nResponda de forma consultiva, prática e sucinta. Se o contexto for insuficiente, peça mais informação."
        answer = llm.answer(prompt)
        return {"result": answer, "retrieved": rows}

    def analise_avancada(state: AgentState) -> AgentState:
        # Interpret some advanced queries and run DuckDB aggregations
        q = (state.pergunta or "").lower()
        # create a working temporary view with normalized column names for SQL convenience
        # map expected columns to normalized ones (create a temp table 'stoppers_work')
        try:
            # Try to produce normalized columns for SQL queries
            # We'll alias user columns to safe names
            df_cols = con.execute(f"DESCRIBE {df_table_name}").fetchdf()
        except Exception:
            pass  # ignore if can't describe
        # Common advanced intents:
        if "resolvid" in q and "prazo" in q or "dentro do prazo" in q or "%" in q or "percent" in q or "percentual" in q:
            # run percent resolved within due date in last month if month mentioned, else overall
            # simple period detection: look for 'mês passado' or 'último mês' or 'ultimo mes' or 'ano'
            # We'll attempt last month
            period = None
            if "mês passado" in q or "mes passado" in q or "último mês" in q or "ultimo mes" in q:
                # compute last month from now
                today = pd.Timestamp.now()
                last = today - pd.DateOffset(months=1)
                period = {"year": last.year, "month": last.month}
            # For duckdb computation, ensure we have aliased columns in a temporary table
            # Let's create a temp view with predictable names from the original df_table_name
            # Mapping best-effort:
            # Try common names variants to rename in the view
            rename_sql = f"""
            CREATE OR REPLACE VIEW stoppers_work AS
            SELECT
                "{get_col_safe(retrieve_original_cols(con, df_table_name), ['Data de solicitacao','Data_de_solicitacao','Data de solicitacao'])}" AS Data_de_solicitacao,
                "{get_col_safe(retrieve_original_cols(con, df_table_name), ['Data de vencimiento','Data_de_vencimiento','Data de vencimiento'])}" AS Data_de_vencimiento,
                "{get_col_safe(retrieve_original_cols(con, df_table_name), ['Data de encerramento','Data_de_encerramento','Data de encerramento'])}" AS Data_de_encerramento,
                *
            FROM {df_table_name}
            """
            # If we can't robustly rename, ignore and just try working with original columns (duckdb is tolerant)
            # Call percent function
            res_text = duckdb_percent_resolved_within(con, "stoppers_work" if view_exists(con, "stoppers_work") else df_table_name, period)
            # Use LLM to polish textual output if needed
            prompt = f"Usuário perguntou: {state.pergunta}\nResultado analítico bruto:\n{res_text}\n\nFormule uma resposta humana e consultiva com recomendações práticas."
            answer = llm.answer(prompt)
            return {"result": answer, "retrieved": None}
        # Other grouping requests: "por criticidade", "por tipo", "por responsavel"
        match = re.search(r"por (\w+)", q)
    if match:
        col = match.group(1)
        # try mapping small Portuguese column words to actual column names heuristically
        mapping = {
            "criticidade": ["Criticidade","criticidade"],
            "tipo": ["Tipo de Stopper","Tipo","Tipo_de_Stopper"],
            "responsavel": ["Responsavel","responsavel","Papel responsable","Fornecedor responsavel"],
            "fornecedor": ["Fornecedor responsavel","fornecedor","Fornecedor"]
        }
        cand_cols = mapping.get(col, [col])
        chosen = get_col_safe(retrieve_original_cols(con, df_table_name), cand_cols, default=col)
        res_text = duckdb_group_count_percent(con, chosen, df_table_name, top_n=6)
        prompt = f"Usuário perguntou: {state.pergunta}\nResultado analítico bruto:\n{res_text}\n\nResponda de forma humana e sugira ações práticas."
        answer = llm.answer(prompt)
        return {"result": answer, "retrieved": None}

    # Fallback generic: run top N stoppers frequency
    res_text = duckdb_group_count_percent(con, '"Nome Stopper"' if column_exists(con, df_table_name, "Nome Stopper") else "Nome Stopper", df_table_name, top_n=6)
    prompt = f"Usuário perguntou: {state.pergunta}\nResultado analítico bruto:\n{res_text}\n\nTransforme em resposta humana e consultiva."
    answer = llm.answer(prompt)
    return {"result": answer, "retrieved": None}

    def pedir_info(state: AgentState) -> AgentState:
        # Ask the user to clarify
        return {"result": "Preciso de mais detalhes para responder; por favor especifique o período, cidade ou coluna desejada.", "retrieved": None}

    def abrir_chamado(state: AgentState) -> AgentState:
        # Provide template for opening ticket
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

    # flow: start -> triagem -> branch
    graph.set_entry_point("triagem")
    graph.add_edge("triagem", "auto_resolver")
    graph.add_edge("triagem", "analise_avancada")
    graph.add_edge("triagem", "abrir_chamado")
    graph.add_edge("triagem", "pedir_info")
    # each node returns END
    graph.add_edge("auto_resolver", END)
    graph.add_edge("analise_avancada", END)
    graph.add_edge("pedir_info", END)
    graph.add_edge("abrir_chamado", END)

    return graph.compile()

# -----------------------
# Helper functions for duckdb column mapping (used in advanced node)
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
        # DuckDB may not support sqlite_master; fallback false
        return False

# -----------------------
# Gradio / main
# -----------------------
def main():
    # Load stoppers
    df = load_stoppers(STOPPERS_CSV)
    # Quick normalize: create consistent internal column names by aliasing in duckdb (we'll register as 'stoppers')
    con = duckdb.connect(database=':memory:')
    # Try simple renames to remove spaces and accents for SQL convenience (but keep original df for RAG)
    safe_df = df.copy()
    # normalize date column names if they exist (map a few possibilities)
    col_map = {}
    for col in safe_df.columns:
        safe_name = col.replace(" ", "_").replace("-", "_").replace(".", "").replace("/","_")
        if safe_name != col:
            safe_df = safe_df.rename(columns={col: safe_name})
            col_map[safe_name] = col
    # Register to duckdb
    con.register("stoppers", safe_df)
    table_name = "stoppers"

    # Build semantic index: choose text columns to include in embeddings
    # We'll include Nome Stopper, Tipo de Stopper, Descrição, Criticidade, Responsavel, Fornecedor responsavel, Nombres de sitios
    candidate_text_cols = [c for c in safe_df.columns if any(k.lower() in c.lower() for k in ["nome", "stopper", "descricao", "descrição", "tipo", "criticidade", "responsavel", "fornecedor", "nombres", "nombres_de"])]
    if not candidate_text_cols:
        candidate_text_cols = safe_df.columns.tolist()
    logger.info(f"Colunas textuais para embeddings: {candidate_text_cols}")
    sem_index = SemanticIndex(safe_df, candidate_text_cols)

    # LLM provider
    llm = LLMProvider()

    # Build LangGraph
    graph = build_graph(sem_index, con, table_name, llm)

    # Conversation memory (very simple, per-session)
    sessions: Dict[str, List[Dict[str,str]]] = {}

    # Gradio UI
    with gr.Blocks(title="Agente Stoppers - Brisanet") as demo:
        gr.Markdown("# Agente Stoppers — Brisanet\nPergunte sobre stoppers. Peça análises avançadas explicitamente (ex.: '% resolvidos dentro do prazo no mês passado').")
        user_input = gr.Textbox(lines=2, label="Sua pergunta")
        session_id = gr.Textbox(value="default", label="Session ID (use para manter contexto)", visible=False)
        chat_output = gr.Textbox(lines=12, label="Resposta")
        btn = gr.Button("Enviar")

        def handle_question(question: str, sess: str):
            sess = sess or "default"
            # store history
            sessions.setdefault(sess, [])
            # invoke graph
            state_in = {"pergunta": question}
            out = graph.invoke(state_in)
            # out is a Pydantic model dict
            result = out.get("result") or "Sem resposta gerada."
            # Append to memory
            sessions[sess].append({"q": question, "a": result})
            return result

        btn.click(fn=handle_question, inputs=[user_input, session_id], outputs=[chat_output])

    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)

if __name__ == "__main__":
    main()