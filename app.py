"""
app.py — RAG + LangGraph + Gradio para bases CSV (Sites, Stoppers, Projeto, Tarefas)
Feito para Railway: usa PORT do env e server_name="0.0.0.0".
Recursos:
 - Cache de embeddings (.npz) por base para inicialização rápida
 - Hybrid retriever: DuckDB (filtros SQL) + busca semântica (SentenceTransformer)
 - LLM via OpenAI ChatCompletion (fallback para mock se OPENAI_API_KEY ausente)
 - LangGraph para orquestração simples
 - Gradio com histórico, exibição de trechos recuperados e scores
 - Logging básico
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import duckdb
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from langgraph.graph import StateGraph, END

# Optional OpenAI import; only used if OPENAI_API_KEY provided
try:
    import openai
except Exception:
    openai = None

# ---------------------------
# Config
# ---------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")  # coloque seus CSVs aqui
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # ajuste se necessário
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "embed_cache")
TOP_K = int(os.getenv("TOP_K", "6"))
PORT = int(os.getenv("PORT", 8080))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Cria diretórios
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_app")

# ---------------------------
# Utils: carregar CSVs
# ---------------------------
def load_bases(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    bases = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV não encontrado: {path} (espere colocar {name})")
        df = pd.read_csv(path, dtype=str).fillna("")
        # Normalize column names
        df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
        # Trim strings
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
        bases[name] = df.reset_index(drop=True)
        logger.info(f"Carregada base {name} com {len(df)} linhas.")
    return bases

# ---------------------------
# Semantic index with cache
# ---------------------------
class SemanticIndex:
    def __init__(self, name: str, df: pd.DataFrame, text_cols: List[str], model_name: str = EMBEDDING_MODEL):
        self.name = name
        self.df = df.reset_index(drop=True)
        self.text_cols = text_cols or df.columns.tolist()
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.corpus = (self.df[self.text_cols].fillna("").astype(str).agg(" \n ".join, axis=1)).tolist()
        self.emb = self._load_or_build_embeddings()
        logger.info(f"Index {self.name}: corpus {len(self.corpus)} embeddings shape {self.emb.shape}")

    def cache_path(self) -> str:
        safe = "".join([c for c in self.name if c.isalnum() or c in "_-"]).lower()
        return os.path.join(EMBED_CACHE_DIR, f"{safe}_emb.npz")

    def _load_or_build_embeddings(self) -> np.ndarray:
        path = self.cache_path()
        if os.path.exists(path):
            try:
                npz = np.load(path)
                emb = npz["emb"]
                if emb.shape[0] == len(self.corpus):
                    logger.info(f"Carregando embeddings em cache para {self.name} ({path})")
                    return emb
                else:
                    logger.info("Cache mismatch de tamanho, recalculando embeddings.")
            except Exception as e:
                logger.warning(f"Erro ao carregar cache {path}: {e}. Recalculando.")
        # Build
        logger.info(f"Calculando embeddings para {self.name} usando {self.model_name} ...")
        emb = self.model.encode(self.corpus, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        np.savez_compressed(path, emb=emb)
        logger.info(f"Embeddings salvos em cache: {path}")
        return emb

    def search(self, query: str, k: int = TOP_K) -> List[Tuple[int, float]]:
        if not isinstance(query, str) or query.strip() == "":
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.emb)[0]
        topk = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in topk]

# ---------------------------
# HybridRetriever
# ---------------------------
class HybridRetriever:
    def __init__(self, bases: Dict[str, pd.DataFrame]):
        self.bases = bases
        # heurística de colunas textuais por base (ajuste para o seu CSV)
        self.text_cols_map = {
            "Sites": [c for c in bases.get("Sites", pd.DataFrame()).columns if c.lower() in {"codigo","estado","tipodesite","observacao","descricao","nome","nome_site"}],
            "Stoppers": [c for c in bases.get("Stoppers", pd.DataFrame()).columns if c.lower() in {"cod_stopper","codigo_stopper","descricao","criticidade","risco","base","status","observacao"}],
            "Projeto": [c for c in bases.get("Projeto", pd.DataFrame()).columns if c.lower() in {"codigo","fase","observacao","descricao","escopo","nome_projeto"}],
            "Tarefas": [c for c in bases.get("Tarefas", pd.DataFrame()).columns if c.lower() in {"codigo","atividade","status","observacao","descricao","executor","responsavel"}],
        }
        # fallback: se a lista ficar vazia, use todas as colunas
        self.indexes = {}
        for name, df in bases.items():
            text_cols = self.text_cols_map.get(name) or df.columns.tolist()
            logger.info(f"Construindo SemanticIndex para base '{name}' com colunas: {text_cols}")
            self.indexes[name] = SemanticIndex(name, df, text_cols)

    def sql_filter(self, base: str, where_sql: str, limit: int = 200) -> pd.DataFrame:
        if base not in self.bases:
            raise KeyError(f"Base desconhecida: {base}")
        df = self.bases[base]
        duckdb.register("tmp", df)
        try:
            q = f"SELECT * FROM tmp WHERE {where_sql} LIMIT {limit}"
            out = duckdb.sql(q).df()
            logger.info(f"SQL filter on {base}: '{where_sql}' -> {len(out)} rows")
            return out
        finally:
            try:
                duckdb.unregister("tmp")
            except Exception:
                pass

    def semantic_search(self, base: str, query: str, k: int = TOP_K) -> pd.DataFrame:
        if base not in self.indexes:
            raise KeyError(f"Base desconhecida: {base}")
        idx = self.indexes[base]
        hits = idx.search(query, k=k)
        rows = []
        for i, score in hits:
            row = idx.df.iloc[i].copy()
            row["_score"] = score
            row["_snippet"] = " / ".join([str(row[col]) for col in idx.text_cols if col in row.index and str(row[col]).strip() != ""])
            rows.append(row)
        df_hits = pd.DataFrame(rows)
        logger.info(f"Semantic search on {base}: query='{query}' -> {len(df_hits)} hits")
        return df_hits

    def combined_search(self, base: str, query: str, where_sql: Optional[str] = None, k: int = TOP_K) -> pd.DataFrame:
        """
        Primeiro filtra com SQL (se provided) e depois roda busca semântica somente sobre o subset.
        Se where_sql for None, roda a busca semântica na base inteira.
        """
        if where_sql:
            filtered = self.sql_filter(base, where_sql, limit=1000)  # subset razoável
            if filtered.empty:
                return filtered
            # montar índice temporário para o subset (não cacheado)
            temp_idx = SemanticIndex(f"{base}_temp", filtered, filtered.columns.tolist(), model_name=EMBEDDING_MODEL)
            hits = temp_idx.search(query, k=k)
            rows = []
            for i, score in hits:
                row = temp_idx.df.iloc[i].copy()
                row["_score"] = score
                row["_snippet"] = " / ".join([str(row[c]) for c in temp_idx.text_cols if str(row[c]).strip() != ""])
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            return self.semantic_search(base, query, k=k)

# ---------------------------
# LLM Provider
# ---------------------------
class LLMProvider:
    def __init__(self, model_name: str = OPENAI_MODEL, openai_key: str = OPENAI_API_KEY):
        self.model_name = model_name
        self.use_openai = False
        if openai_key and openai is not None:
            openai.api_key = openai_key
            self.use_openai = True
            logger.info("LLMProvider: usando OpenAI ChatCompletion")
        else:
            if openai_key and openai is None:
                logger.warning("OPENAI_API_KEY presente mas pacote 'openai' não instalado. Usando fallback mock.")
            else:
                logger.info("OPENAI_API_KEY não setada. Usando LLM mock (local) para testes.")
            self.use_openai = False

    def chat(self, query: str, context: str) -> str:
        """
        Faz prompt para o modelo. Se OpenAI não estiver disponível, retorna uma resposta estruturada mock.
        A resposta inclui: 1) trechos usados (context preview), 2) resposta final (resumo).
        """
        system = (
            "Você é um assistente técnico especializado em implantação de redes (ex.: Sites, Stoppers, Tarefas, Projeto). "
            "Responda com objetividade, cite quais trechos dos dados usou e indique possíveis incertezas."
        )
        prompt = f"""
Contexto recuperado (trechos relevantes):
{context}

Pergunta:
{query}

Instruções:
1) Primeiro liste os 3 principais achados do contexto (breve).
2) Em seguida, proponha a resposta direta à pergunta (até 200 palavras).
3) Se houver incerteza ou falta de dados, diga explicitamente o que falta.
4) Se aplicável, indique qual base (Sites, Stoppers, Projeto, Tarefas) parece conter as informações.
"""
        if self.use_openai:
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0
                )
                text = resp["choices"][0]["message"]["content"].strip()
                return text
            except Exception as e:
                logger.error(f"Erro ao chamar OpenAI: {e}")
                # fallback to mock formatting
        # Mock fallback
        snippet_preview = "\n".join(context.splitlines()[:6])
        return (
            f"--- MOCK RESPONDER (OpenAI não configurado) ---\n\n"
            f"Trechos usados (preview):\n{snippet_preview}\n\n"
            f"Resposta (mock):\nBaseado nos trechos acima, resposta direta para: {query}\n\n"
            f"(Configure OPENAI_API_KEY para usar o modelo real.)"
        )

# ---------------------------
# LangGraph pipeline
# ---------------------------
class AgentState(BaseModel):
    query: str
    base: str
    where_sql: Optional[str] = None
    k: int = TOP_K
    result: Optional[str] = None
    retrieved: Optional[List[Dict[str, Any]]] = None

def build_graph(retriever: HybridRetriever, llm: LLMProvider):
    graph = StateGraph(AgentState)

    def retrieve(state: AgentState) -> AgentState:
        # Decide whether usar combined_search (SQL + sem)
        if state.where_sql and state.where_sql.strip() != "":
            df = retriever.combined_search(state.base, state.query, where_sql=state.where_sql, k=state.k)
        else:
            df = retriever.semantic_search(state.base, state.query, k=state.k)
        if df is None or df.empty:
            state.retrieved = []
            context = ""
        else:
            # Monta contexto com top N trechos e scores
            rows = df.to_dict(orient="records")
            state.retrieved = rows
            # Build plain text context: include snippet and score and selected columns
            parts = []
            for r in rows:
                score = r.get("_score", "")
                snippet = r.get("_snippet", "")
                # keep some key columns to help the LLM
                key_info = []
                for kcol in ["codigo", "cod_stopper", "atividade", "nome", "estado", "fase", "executor"]:
                    if kcol in r and str(r[kcol]).strip() != "":
                        key_info.append(f"{kcol}: {r[kcol]}")
                parts.append(f"SCORE={score:.4f} | {'; '.join(key_info)}\nSNIPPET: {snippet}")
            context = "\n\n".join(parts)
        ans = llm.chat(state.query, context)
        state.result = ans
        return state

    graph.add_node("retrieve", retrieve)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()

# ---------------------------
# Gradio UI
# ---------------------------
def main():
    # Paths esperados (coloque seus CSVs em data/)
    expected = {
        "Sites": os.path.join(DATA_DIR, "sites.csv"),
        "Stoppers": os.path.join(DATA_DIR, "stoppers.csv"),
        "Projeto": os.path.join(DATA_DIR, "projetos.csv"),
        "Tarefas": os.path.join(DATA_DIR, "tarefas.csv"),
    }
    # Validar arquivos
    missing = [p for p in expected.values() if not os.path.exists(p)]
    if missing:
        logger.warning("Arquivos faltando em data/: " + ", ".join(missing))
        # Não falha aqui para facilitar testes locais; mas avisa o usuário
    # Carrega só os CSVs existentes
    present = {k: v for k, v in expected.items() if os.path.exists(v)}
    if not present:
        # Para permitir testes locais sem dados, cria bases vazias com colunas mínimas
        logger.info("Nenhuma base encontrada em data/ — criando bases vazias para UI.")
        bases = {
            "Sites": pd.DataFrame(columns=["codigo", "nome", "estado", "observacao"]),
            "Stoppers": pd.DataFrame(columns=["cod_stopper", "descricao", "criticidade", "risco", "observacao"]),
            "Projeto": pd.DataFrame(columns=["codigo", "fase", "descricao", "observacao"]),
            "Tarefas": pd.DataFrame(columns=["codigo", "atividade", "status", "observacao", "executor"]),
        }
    else:
        bases = load_bases({k: expected[k] for k in expected if os.path.exists(expected[k])})

    retriever = HybridRetriever(bases)
    llm = LLMProvider()
    graph = build_graph(retriever, llm)

    # Gradio components
    base_choices = list(bases.keys())
    dropdown = gr.Dropdown(base_choices, label="Base", value=base_choices[0] if base_choices else None)
    query_box = gr.Textbox(lines=2, label="Pergunta (ex.: 'Quais stoppers com criticidade alta aguardam execução?')")
    where_box = gr.Textbox(lines=1, label="Filtro SQL opcional (ex.: estado='CE' )", placeholder="Deixe vazio para sem filtro")
    k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=TOP_K, label="Número de trechos recuperados (k)")

    # Outputs: resposta textual + trechos recuperados (tabela)
    response_out = gr.Textbox(lines=15, label="Resposta do Agente")
    retrieved_out = gr.Dataframe(headers=["_snippet", "_score"], label="Trechos Recuperados (snippet + score)")

    def run_agent(base, query, where_sql, k):
        t0 = time.time()
        state = AgentState(query=query or "", base=base or "", where_sql=(where_sql or None), k=int(k))
        out_state = graph.invoke(state)
        # prepare retrieved table
        if out_state.retrieved:
            df = pd.DataFrame(out_state.retrieved)
            # ensure snippet and score are present
            if "_snippet" not in df.columns:
                df["_snippet"] = df.apply(lambda r: " / ".join([str(r[c]) for c in r.index if c not in ["_score"] and str(r[c]).strip() != ""]), axis=1)
            if "_score" not in df.columns:
                df["_score"] = ""
            # reduce columns for display
            display_df = df[["_snippet", "_score"]].copy()
        else:
            display_df = pd.DataFrame([{"_snippet": "Nenhum trecho encontrado", "_score": ""}])
        elapsed = time.time() - t0
        logger.info(f"Pergunta='{query}' Base={base} k={k} -> tempo {elapsed:.2f}s")
        return out_state.result or "Sem resposta", display_df

    with gr.Blocks(title="Agente RAG - Brisanet (Sites/Stoppers/Projeto/Tarefas)") as demo:
        gr.Markdown("# Agente RAG — Rápido e robusto")
        with gr.Row():
            with gr.Column(scale=3):
                dropdown.render()
                query_box.render()
                where_box.render()
                k_slider.render()
                btn = gr.Button("Enviar")
            with gr.Column(scale=2):
                response_out.render()
                retrieved_out.render()
        btn.click(fn=run_agent, inputs=[dropdown, query_box, where_box, k_slider], outputs=[response_out, retrieved_out])

    # Launch
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)

if __name__ == "__main__":
    main()
