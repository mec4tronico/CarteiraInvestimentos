"""app_carteira.py
# Teste de edição - Comentário inserido com sucesso!

Dashboard Interativo da Carteira de Investimentos
Ajustado com reordenação personalizada de colunas e regras de custódia B3.
""

from concurrent.futures import ThreadPoolExecutor
import json
import time
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen
import sqlite3
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Configuração da Página no Streamlit
st.set_page_config(
    page_title="Minha Carteira B3 - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded
)

# ==============================================================================
# CONFIGURAÇÕES E EXCEÇÕES DE REGRAS DE NEGÓCIO
# ==============================================================================
TICKER_MAP = {
    "ELET3": "AXIA3",
    "ELET6": "AXIA6",
    # Adicione novos de-para aqui caso outros ativos mudem de código
}

# Componentes do IFIX em 01/08/2026. A composição é revisada
# quadrimestralmente; atualize esta lista após cada revisão da B3.
# Ativos fora desta lista são tratados como Ação, evitando que Units e ETFs
# terminados em 11 (como SANB11 e BOVA11) sejam classificados como FIIs.
TICKERS_FII_IFIX = {
    'AFHI11', 'ALZR11', 'AZPL11', 'BBIG11', 'BCRI11', 'BCIA11', 'BPML11',
    'BRCO11', 'BRCR11', 'BROF11', 'BTAL11', 'BTLG11', 'BTHF11', 'BTCI11',
    'CACR11', 'CLIN11', 'CPSH11', 'CPTS11', 'CYCR11', 'DEVA11', 'FATN11',
    'GARE11', 'GGRC11', 'GTWR11', 'GZIT11', 'HABT11', 'HCTR11', 'HFOF11',
    'HGBS11', 'HGCR11', 'HGRE11', 'HGLG11', 'HGRU11', 'HSAF11', 'HSLG11',
    'HSML11', 'HTMX11', 'ICRI11', 'IRIM11', 'ITRI11', 'JSAF11', 'JSRE11',
    'KCRE11', 'KFOF11', 'KISU11', 'KIVO11', 'KNCR11', 'KNHF11', 'KNHY11',
    'KNIP11', 'KNRI11', 'KNSC11', 'KNUQ11', 'KORE11', 'LIFE11', 'LVBI11',
    'MANA11', 'MCCI11', 'MCRE11', 'MFII11', 'MXRF11', 'OUJP11', 'PCIP11',
    'PMLL11', 'PORD11', 'PSEC11', 'PVBI11', 'RBFM11', 'RBRL11', 'RBRP11',
    'RBRR11', 'RBRX11', 'RBRY11', 'RBVA11', 'RCRB11', 'RECR11', 'RPRI11',
    'RZAK11',
    'RZAT11', 'RZTR11', 'SNCI11', 'SNEL11', 'SNFF11', 'SPXS11', 'TEPP11',
    'TGAR11', 'TOPP11', 'TRBL11', 'TRXF11', 'TVRI11', 'URPR11', 'VCJR11',
    'VGHF11', 'VGIP11', 'VGIR11', 'VGRI11', 'VILG11', 'VINO11', 'VISC11',
    'VRTA11', 'VRTM11', 'WHGR11', 'XPCI11', 'XPLG11', 'XPSF11', 'XPML11',
}

# Tolerância Mínima: Ativos com quantidade <= QTD_MINIMA_TOLERANCIA serão descartados
QTD_MINIMA_TOLERANCIA = 5.0

# ==============================================================================
# SQLITE (ETAPA 1) — INFRAESTRUTURA, TABELAS E FUNÇÕES DE DADOS
# ==============================================================================
def _resolver_db_path() -> str:
    """Resolve o caminho do banco garantindo compatibilidade com dados antigos.

    - Se existir 'dados/carteira.db', mantém compatibilidade usando esse arquivo.
    - Caso contrário, usa 'carteira.db' na raiz (conforme especificação).
    """
    caminho_legado = os.path.join('dados', 'carteira.db')
    if os.path.exists(caminho_legado):
        return caminho_legado
    return 'carteira.db'

DB_PATH = _resolver_db_path()

def conectar_banco() -> sqlite3.Connection:
    """Conecta ao SQLite e retorna uma conexão configurada.

    - Habilita foreign_keys.
    - Usa row_factory = sqlite3.Row.
    - O chamador deve fechar a conexão.
    """
    try:
        pasta = os.path.dirname(DB_PATH)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
    except Exception:
        # Se falhar a criação de pasta, segue (pode ser DB_PATH na raiz)
        pass

    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
    except Exception:
        # Se o ambiente restringir algum pragma, segue normalmente
        pass
    return conn

def criar_banco() -> None:
    """Cria automaticamente o arquivo do banco caso não exista e garante tabelas."""
    try:
        conn = conectar_banco()
        try:
            criar_tabelas(conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        # Evita quebrar o app: sem banco, o dashboard ainda deve funcionar.
        pass

def criar_tabelas(conn: sqlite3.Connection) -> None:
    """Cria as tabelas e índices necessários (ETAPA 1)."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ativos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            nome TEXT,
            tipo TEXT,
            quantidade REAL NOT NULL DEFAULT 0,
            preco_medio REAL NOT NULL DEFAULT 0,
            data_criacao TEXT
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movimentacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            tipo_movimentacao TEXT,
            quantidade REAL,
            preco REAL,
            valor_total REAL,
            corretagem REAL,
            impostos REAL,
            data TEXT,
            observacoes TEXT
        );
    "")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dividendos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            data_pagamento TEXT,
            valor REAL,
            quantidade REAL,
            valor_total REAL
        );
    "")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS configuracoes (
            chave TEXT PRIMARY KEY,
            valor TEXT
        );
    """)

    # Índices para performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ativos_ticker ON ativos(ticker);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_movimentacoes_ticker_data ON movimentacoes(ticker, data);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_movimentacoes_data ON movimentacoes(data);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dividendos_ticker_data ON dividendos(ticker, data_pagamento);")

    conn.commit()

def _executar_em_transacao(conn: sqlite3.Connection, func):
    """Executa uma função dentro de transação, com tratamento de exceções."""
    try:
        conn.execute("BEGIN;")
        retorno = func()
        conn.commit()
        return retorno
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise

def inserir_movimentacao(
    ticker: str,
    tipo_movimentacao: str,
    quantidade: float,
    preco: float,
    valor_total: float,
    corretagem: float,
    impostos: float,
    data: datetime,
    observacoes: str = ""
) -> int:
    """Insere uma movimentação na tabela `movimentacoes` usando transação."""
    if ticker is None or str(ticker).strip() == "":
        return 0

    ticker_norm = str(ticker).strip().upper()
    tipo_mov = str(tipo_movimentacao).strip() if tipo_movimentacao is not None else ""
    obs = str(observacoes).strip() if observacoes is not None else ""
    data_str = None
    try:
        data_str = pd.to_datetime(data).strftime("%Y-%m-%d")
    except Exception:
        data_str = None

    def _do_insert():
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO movimentacoes
            (ticker, tipo_movimentacao, quantidade, preco, valor_total, corretagem, impostos, data, observacoes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                ticker_norm,
                tipo_mov,
                float(quantidade) if quantidade is not None else None,
                float(preco) if preco is not None else None,
                float(valor_total) if valor_total is not None else None,
                float(corretagem) if corretagem is not None else None,
                float(impostos) if impostos is not None else None,
                data_str,
                obs
            )
        )
        return int(cursor.lastrowid or 0)

    conn = None
    try:
        conn = conectar_banco()
        criar_tabelas(conn)
        return _executar_em_transacao(conn, _do_insert)
    except Exception:
        return 0
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def editar_movimentacao(
    id_movimentacao: int,
    ticker: str,
    tipo_movimentacao: str,
    quantidade: float,
    preco: float,
    valor_total: float,
    corretagem: float,
    impostos: float,
    data: datetime,
    observacoes: str = ""
) -> bool:
    ""Edita uma movimentação existente."""
    if not id_movimentacao:
        return False

    ticker_norm = str(ticker).strip().upper() if ticker is not None else ""
    tipo_mov = str(tipo_movimentacao).strip() if tipo_movimentacao is not None else ""
    obs = str(observacoes).strip() if observacoes is not None else ""
    data_str = None
    try:
        data_str = pd.to_datetime(data).strftime("%Y-%m-%d")
    except Exception:
        data_str = None

    def _do_update():
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE movimentacoes
               SET ticker = ?,
                   tipo_movimentacao = ?,
                   quantidade = ?,
                   preco = ?,
                   valor_total = ?,
                   corretagem = ?,
                   impostos = ?,
                   data = ?,
                   observacoes = ?
             WHERE id = ?;
            """,
            (
                ticker_norm,
                tipo_mov,
                float(quantidade) if quantidade is not None else None,
                float(preco) if preco is not None else None,
                float(valor_total) if valor_total is not None else None,
                float(corretagem) if corretagem is not None else None,
                float(impostos) if impostos is not None else None,
                data_str,
                obs,
                int(id_movimentacao)
            )
        )
        return cursor.rowcount > 0

    conn = None
    try:
        conn = conectar_banco()
        criar_tabelas(conn)
        return bool(_executar_em_transacao(conn, _do_update))
    except Exception:
        return False
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def excluir_movimentacao(id_movimentacao: int) -> bool:
    """Exclui uma movimentação por ID."""
    if not id_movimentacao:
        return False

    def _do_delete():
        cursor = conn.cursor()
        cursor.execute("DELETE FROM movimentacoes WHERE id = ?;", (int(id_movimentacao),))
        return cursor.rowcount > 0

    conn = None
    try:
        conn = conectar_banco()
        criar_tabelas(conn)
        return bool(_executar_em_transacao(conn, _do_delete))
    except Exception:
        return False
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def consultar_movimentacoes(ticker: str = None) -> pd.DataFrame:
    """Consulta movimentações, opcionalmente filtrando por ticker."""
    conn = None
    try:
        conn = conectar_banco()
        criar_tabelas(conn)

        if ticker:
            ticker_norm = str(ticker).strip().upper()
            df = pd.read_sql_query(
                """
                SELECT id, ticker, tipo_movimentacao, quantidade, preco, valor_total, corretagem, impostos, data, observacoes
                  FROM movimentacoes
                 WHERE ticker = ?
                 ORDER BY data ASC, id ASC;
                """,
                conn,
                params=(ticker_norm,)
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT id, ticker, tipo_movimentacao, quantidade, preco, valor_total, corretagem, impostos, data, observacoes
                  FROM movimentacoes
                 ORDER BY data ASC, id ASC;
                """,
                conn
            )

        # Normalizações importantes para compatibilidade com o cálculo
        if not df.empty:
            df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
            df["data_dt"] = pd.to_datetime(df["data"], errors="coerce")
        else:
            df["data_dt"] = pd.to_datetime(pd.Series(dtype=datetime64[ns]"))

        return df
    except Exception:
        return pd.DataFrame()
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def calcular_preco_medio(custo_total: float, quantidade: float) -> float:
    """Calcula preço médio (mantido simples para uso interno)."""
    try:
        if quantidade and quantidade > 0:
            return float(custo_total) / float(quantidade)
        return 0.0
    except Exception:
        return 0.0

def calcular_patrimonio(preco_atual: float, quantidade: float) -> float:
    """Calcula patrimônio (mantido simples para uso interno)."""
    try:
        return float(preco_atual) * float(quantidade)
    except Exception:
        return 0.0

def calcular_dividendos(ticker: str = None) -> pd.DataFrame:
    """Consulta dividendos armazenados no banco (placeholder para compatibilidade da Etapa 1)."""
    conn = None
    try:
        conn = conectar_banco()
        criar_tabelas(conn)

        if ticker:
            ticker_norm = str(ticker).strip().upper()
            df = pd.read_sql_query(
                """
                SELECT id, ticker, data_pagamento, valor, quantidade, valor_total
                  FROM dividendos
                 WHERE ticker = ?
                 ORDER BY data_pagamento ASC, id ASC;
                """,
                conn,
                params=(ticker_norm,)
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT id, ticker, data_pagamento, valor, quantidade, valor_total
                  FROM dividendos
                 ORDER BY data_pagamento ASC, id ASC;
                """,
                conn
            )

        if not df.empty:
            df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
            df["data_pagamento_dt"] = pd.to_datetime(df["data_pagamento"], errors="coerce")
        else:
            df["data_pagamento_dt"] = pd.to_datetime(pd.Series(dtype="datetime64[ns]"))
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def _upsert_ativos_resumo(df_resumo: pd.DataFrame) -> None:
    """Sincroniza a tabela `ativos` com o resumo atual.

    Mantém compatibilidade e performance: substitui os registros por ticker.
    """
    if df_resumo is None or df_resumo.empty:
        return

    conn = None
    try:
        conn = conectar_banco()
        criar_tabelas(conn)

        def _do_upsert():
            cursor = conn.cursor()
            agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for _, row in df_resumo.iterrows():
                ticker = str(row.get("ticker", "")).strip().upper()
                if ticker == "":
                    continue
                tipo = str(row.get("tipo_ativo", "")).strip()
                quantidade = float(row.get("quantidade", 0.0) or 0.0)
                preco_medio = float(row.get("preco_medio", 0.0) or 0.0)
                cursor.execute(
                    """
                    INSERT INTO ativos (ticker, nome, tipo, quantidade, preco_medio, data_criacao)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        nome = excluded.nome,
                        tipo = excluded.tipo,
                        quantidade = excluded.quantidade,
                        preco_medio = excluded.preco_medio;
                    """,
                    (ticker, None, tipo, quantidade, preco_medio, agora)
                )

        # Garantir que exista uma constraint UNIQUE para ON CONFLICT funcionar
        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uidx_ativos_ticker ON ativos(ticker);")
            conn.commit()
        except Exception:
            pass

        _executar_em_transacao(conn, _do_upsert)

    except Exception:
        pass
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def _inserir_movimentacoes_lote(df_ops: pd.DataFrame) -> int:
    """Insere em lote as movimentações no SQLite com transações.

    Mantém compatibilidade com importações repetidas: não tenta deduplicar por regra
    de negócio (porque isso poderia alterar comportamento). A deduplicação pode ser
    evoluída em etapas posteriores mantendo a lógica original.
    """
    if df_ops is None or df_ops.empty:
        return 0

    conn = None
    try:
        conn = conectar_banco()
        criar_tabelas(conn)

        def _do_insert():
            cursor = conn.cursor()
            inseridos = 0
            for _, row in df_ops.iterrows():
                ticker = str(row.get("Ticker", "")).strip().upper()
                if ticker == "":
                    continue

                tipo_mov = str(row.get("Movimentação", "")).strip()
                qtd = float(row.get("Quantidade_num", 0.0) or 0.0)
                preco = float(row.get("Preco_num", 0.0) or 0.0)
                valor_total = float(row.get("Valor_num", 0.0) or 0.0)

                data_dt = row.get("Data_dt", None)
                data_str = None
                try:
                    data_str = pd.to_datetime(data_dt).strftime("%Y-%m-%d")
                except Exception:
                    data_str = None

                # Observações: preserva produto original e a marcação de entrada/saída para auditoria
                produto = str(row.get("Produto", "")).strip()
                entrada_saida = str(row.get("Entrada/Saída, "")).strip()
                observacoes = f"Produto: {produto} | Entrada/Saída: {entrada_saida}".strip()

                # Corretagem e impostos não existem no layout atual utilizado; mantém compatibilidade com colunas.
                corretagem = 0.0
                impostos = 0.0

                cursor.execute(
                    """
                    INSERT INTO movimentacoes
                    (ticker, tipo_movimentacao, quantidade, preco, valor_total, corretagem, impostos, data, observacoes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (ticker, tipo_mov, qtd, preco, valor_total, corretagem, impostos, data_str, observacoes)
                )
                inseridos += 1
            return inseridos

        return int(_executar_em_transacao(conn, _do_insert) or 0)

    except Exception:
        return 0
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def calcular_posicao() -> pd.DataFrame:
    """Calcula a posição atual usando o SQLite como origem, preservando a lógica existente."""
    df_db = consultar_movimentacoes()
    if df_db is None or df_db.empty:
        return pd.DataFrame([])

    # Reconstrói o formato interno esperado pela lógica existente
    # - Mantém nomes e regras para não alterar cálculos
    df = df_db.copy()
    df["Ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Extrai Entrada/Saída das observações (compatibilidade), mas preserva fallback seguro
    if observacoes" in df.columns:
        obs = df["observacoes"].astype(str)
        df["Entrada/Saída"] = obs.str.extract(r"Entrada/Saída:\s*([^|]+)(r"Produto:\s*([^|]+)\s*\|", expand=False).fillna(df["Ticker"])
    else:
        df["Entrada/Saída"] = ""
        df["Produto"] = df["Ticker]

    df["Movimentação"] = df.get("tipo_movimentacao", "").astype(str)
    df["Quantidade_num"] = pd.to_numeric(df.get("quantidade", 0.0), errors="coerce").fillna(0.0)
    df["Valor_num"] = pd.to_numeric(df.get("valor_total", 0.0), errors=coerce").fillna(0.0)
    df["Preco_num"] = pd.to_numeric(df.get("preco", 0.0), errors="coerce").fillna(0.0)
    df["Data_dt"] = pd.to_datetime(df.get("data_dt"), errors="coerce)

    # Ordena cronologicamente (da movimentação mais antiga para a mais recente)
    df = df.sort_values(by="Data_dt", ascending=True)

    posicoes = {}

    # Eventos considerados compras/entradas de ativos
    eventos_compra = [
        'Transferência - Liquidação',
        'Compra',
        'ENTRADA EM CUSTODIA S/FINANC.',
        'Bonificação em Ativos',
        'Fração em Ativos'
    ]

    for _, row in df.iterrows():
        ticker = row['Ticker']
        mov = str(row['Movimentação']).strip()
        tipo = str(row['Entrada/Saída']).strip()
        qtd = float(row['Quantidade_num'] or 0.0)
        valor_op = float(row['Valor_num'] or 0.0)
        data_op = row['Data_dt']

        # Filtra ativos inválidos ou títulos de Renda Fixa/Caixa
        if pd.isna(ticker) or ticker == "" or any(rf in ticker for rf in ['CDB', 'CFA', 'RDB', 'Tesouro']):
            continue

        if ticker not in posicoes:
            posicoes[ticker] = {
                'ticker': ticker,
                'quantidade': 0.0,
                'custo_total': 0.0,
                # A data só é definida quando uma posição é efetivamente aberta.
                # Isso evita que eventos anteriores (ou uma posição já encerrada)
                # contaminem a data de aquisição do lote atual.
                'primeira_compra': None,
                # A lista IFIX identifica os FIIs mais líquidos, sem confundir
                # Units e ETFs terminados em 11 com fundos imobiliários.
                'tipo_ativo': classificar_tipo_ativo(row.get('Produto', ticker), ticker),
                # Aportes ainda presentes na posição, para o benchmark CDI.
                'aportes': [],
            }

        # Identifica se é Entrada/Compra com valor financeiro.
        # No relatório de Negociações, Compra/Venda são informados diretamente.
        is_compra = (mov in eventos_compra and tipo == 'Credito')
        # Identifica se é Saída/Venda
        is_venda = (mov == 'Venda' or (mov == 'Transferência - Liquidação' and tipo == 'Debito'))

        if is_compra:
            if valor_op == 0 and float(row.get('Preco_num', 0.0) or 0.0) > 0:
                valor_op = float(row.get('Preco_num', 0.0) or 0.0) * qtd

            posicoes[ticker]['quantidade'] += qtd
            posicoes[ticker]['custo_total'] += valor_op
            posicoes[ticker]['aportes'].append({
                'data': pd.to_datetime(data_op).normalize() if pd.notna(data_op) else pd.Timestamp.today().normalize(),
                'quantidade': qtd,
                'valor': valor_op,
            })

            if posicoes[ticker]['primeira_compra'] is None and pd.notna(data_op):
                posicoes[ticker]['primeira_compra'] = data_op

        elif is_venda and posicoes[ticker]['quantidade'] > 0:
            quantidade_anterior = posicoes[ticker]['quantidade']
            pm = posicoes[ticker]['custo_total'] / posicoes[ticker]['quantidade']
            quantidade_vendida = min(qtd, quantidade_anterior)
            fator_remanescente = 1 - (quantidade_vendida / quantidade_anterior)
            posicoes[ticker]['quantidade'] = max(0.0, quantidade_anterior - qtd)
            posicoes[ticker]['custo_total'] = posicoes[ticker]['quantidade'] * pm

            # Como o preço médio já é usado no saldo da aplicação, a venda reduz
            # proporcionalmente os aportes que permanecem no benchmark CDI.
            for aporte in posicoes[ticker]['aportes']:
                aporte['quantidade'] *= fator_remanescente
                aporte['valor'] *= fator_remanescente
            posicoes[ticker]['aportes'] = [
                aporte for aporte in posicoes[ticker]['aportes']
                if aporte['quantidade'] > 0
            ]

            # Uma venda/transferência de liquidação que zera a posição encerra o lote.
            # A próxima entrada passa a ter sua própria data de aquisição.
            if posicoes[ticker]['quantidade'] == 0:
                posicoes[ticker]['primeira_compra'] = None

    data_final = pd.Timestamp.today().normalize()
    datas_aportes = [
        aporte['data']
        for posicao in posicoes.values()
        for aporte in posicao['aportes']
    ]
    if datas_aportes:
        cdi_diario = buscar_cdi_diario(
            min(datas_aportes).strftime('%d/%m/%Y'),
            data_final.strftime('%d/%m/%Y')
        )
    else:
        cdi_diario = pd.Series(dtype=float)

    # ==============================================================================
    # AUDITORIA DE SALDO: TOLERÂNCIA DE QUANTIDADE > 5
    # ==============================================================================
    resumo = []
    for t, p in posicoes.items():
        if p['quantidade'] > QTD_MINIMA_TOLERANCIA and p['custo_total'] > 0:
            pm = p['custo_total'] / p['quantidade']
            data_exibicao = p['primeira_compra'].strftime('%d/%m/%Y') if pd.notnull(p['primeira_compra']) else '-'
            resumo.append({
                'ticker': p['ticker'],
                'quantidade': p['quantidade'],
                'preco_medio': pm,
                'custo_total': p['custo_total'],
                'valor_equivalente_cdi': calcular_valor_equivalente_cdi(
                    p['aportes'], cdi_diario
                ),
                'data_aquisicao': data_exibicao,
                'tipo_ativo': p['tipo_ativo'],
            })

    df_resumo = pd.DataFrame(resumo)

    # Mantém o banco coerente com o estado calculado (sem interferir na UI)
    try:
        _upsert_ativos_resumo(df_resumo)
    except Exception:
        pass

    return df_resumo

# Inicialização do banco ao carregar o app
criar_banco()

def limpar_valor_numerico(val):
    """Converte valores no formato B3/Excel para float."""
    if pd.isna(val) or val == "-" or str(val).strip() == "":
        return 0.0
    try:
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).replace(".", "").replace(",", ".")
        return float(val_str)
    except Exception:
        return 0.0

def classificar_tipo_ativo(produto, ticker) -> str:
    ""Classifica ativos. Se termina em 11 e não for FII, é Unit (Ação)."""
    descricao = str(produto).upper()
    ticker = str(ticker).upper().strip()
    
    # É FII se estiver na lista IFIX ou tiver "FII" na descrição
    if ticker in TICKERS_FII_IFIX or "FII" in descricao or "IMOBILIARIO" in descricao:
        return "FII"
    
    # Se termina em 11 e não caiu na regra acima, é Unit de Ação (Ex: SAPR11)
    return Ação"

@st.cache_data(ttl=86400, show_spinner=False)
def buscar_cdi_diario(data_inicial: str, data_final: str) -> pd.Series:
    ""Busca o CDI diário (série 12) no Banco Central do Brasil."""
    parametros = urlencode({
        'formato': 'json',
        'dataInicial': data_inicial,
        'dataFinal': data_final,
    })
    url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?{parametros}'

    try:
        with urlopen(url, timeout=15) as resposta:
            dados = json.load(resposta)
        df_cdi = pd.DataFrame(dados)
        df_cdi['data'] = pd.to_datetime(df_cdi['data'], format='%d/%m/%Y')
        df_cdi['valor'] = pd.to_numeric(
            df_cdi['valor'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
        return df_cdi.dropna().set_index('data')['valor'].sort_index()
    except Exception:
        # Sem CDI, o dashboard continua disponível e mostra '-' nas colunas.
        return pd.Series(dtype=float)

def calcular_valor_equivalente_cdi(aportes: list, cdi_diario: pd.Series) -> float:
    """Capitaliza cada aporte remanescente pelo CDI diário até a data atual."""
    if not aportes or cdi_diario.empty:
        return np.nan

    valor_equivalente = 0.0
    for aporte in aportes:
        taxas = cdi_diario[cdi_diario.index >= aporte['data']]
        fator = (1 + taxas / 100).prod()
        valor_equivalente += aporte['valor'] * fator
    return valor_equivalente

@st.cache_data(ttl=1800)
def _ler_planilha_b3(file_bytes) -> pd.DataFrame:
    """Lê e normaliza a planilha B3 para o formato interno de operações.

    A B3 usa nomes de colunas distintos nos dois relatórios. Ambos são
    convertidos para um formato interno único antes do cálculo das posições.
    """
    df = pd.read_excel(file_bytes)

    colunas_negociacoes = {
        'Data do Negócio', 'Tipo de Movimentação', 'Código de Negociação',
        'Quantidade', 'Preço', 'Valor'
    }
    eh_relatorio_negociacoes = colunas_negociacoes.issubset(df.columns)

    if eh_relatorio_negociacoes:
        # Relatório B3 "Negociações": cada linha já representa uma compra ou venda.
        df = df.rename(columns={
            'Data do Negócio': 'Data',
            'Tipo de Movimentação': 'Movimentação',
            'Código de Negociação': 'Ticker',
            'Preço': 'Preço unitário',
            'Valor': 'Valor da Operação',
        })
        # No relatório de Negociações, o sufixo F identifica operações no
        # mercado fracionário. Elas pertencem ao mesmo ativo no mercado padrão
        # (por exemplo, VALE3F e VALE3 são consolidados como VALE3).
        df['Ticker'] = (
            df['Ticker'].astype(str).str.strip().str.upper()
            .str.replace(r'F = df['Ticker']
        df['Entrada/Saída'] = df['Movimentação'].map({
            'Compra': 'Credito',
            'Venda': 'Debito',
        }).fillna('')
    else:
        colunas_movimentacoes = {
            'Data', 'Movimentação', 'Entrada/Saída', 'Produto', 'Quantidade',
            'Preço unitário', 'Valor da Operação'
        }
        faltantes = colunas_movimentacoes - set(df.columns)
        if faltantes:
            raise ValueError(
                'Formato de planilha B3 não reconhecido. Colunas ausentes: '
                + ', '.join(sorted(faltantes))
            )
        # Extrato de Movimentações: o código vem no início da descrição do produto.
        df['Ticker'] = df['Produto'].astype(str).str.split('-').str[0].str.strip()

    # Ordena cronologicamente (da movimentação mais antiga para a mais recente)
    df['Data_dt'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
    df = df.sort_values(by='Data_dt', ascending=True)

    df['Quantidade_num'] = df['Quantidade'].apply(limpar_valor_numerico)
    df['Valor_num'] = df['Valor da Operação'].apply(limpar_valor_numerico)
    df['Preco_num'] = df['Preço unitário'].apply(limpar_valor_numerico)

    # Mapeamento e substituição de tickers antigos (Ex: ELET3 -> AXIA3)
    df['Ticker'] = df['Ticker'].replace(TICKER_MAP)

    return df

def processar_movimentacoes_b3(file_bytes) -> pd.DataFrame:
    """Processa os extratos de Movimentações ou Negociações da B3.

    IMPORTANTE:
    - Mantém a mesma lógica de cálculo existente.
    - A origem dos dados passa a ser o SQLite:
      1) Lê a planilha -> normaliza operações
      2) Insere operações no SQLite
      3) Recalcula a posição consultando o SQLite
    """
    df_ops = _ler_planilha_b3(file_bytes)

    # Insere as movimentações no banco (transação)
    try:
        _inserir_movimentacoes_lote(df_ops)
    except Exception:
        # Sem quebrar o app: se falhar a gravação, ainda tenta calcular com o que já existe no banco
        pass

    # A partir daqui, a posição é sempre recalculada a partir do banco
    return calcular_posicao()

def buscar_dados_yfinance(ticker_b3: str) -> dict:
    """Busca cotações atuais e métricas via Yahoo Finance para ativos da B3."""
    symbol = f"{ticker_b3}.SA" if not ticker_b3.endswith(".SA") else ticker_b3
    yticker = yf.Ticker(symbol)

    try:
        info = yticker.info or {}
    except Exception:
        info = {}

    try:
        hist = yticker.history(period="1mo")
    except Exception:
        hist = pd.DataFrame()

    preco_atual = info.get("regularMarketPrice) or info.get("currentPrice") or info.get("previousClose")
    if (preco_atual is None or preco_atual == 0) and not hist.empty:
        preco_atual = float(hist["Close"].iloc[-1])

    dy = None
    try:
        dividends = yticker.dividends
        if dividends is not None and not dividends.empty and preco_atual and preco_atual > 0:
            um_ano_atras = pd.Timestamp.now(tz=dividends.index.tz) - pd.Timedelta(days=365)
            divs_12m = dividends[dividends.index >= um_ano_atras].sum()
            dy = (divs_12m / preco_atual) * 100
    except Exception:
        dy = None

    if dy is None or dy == 0:
        dy_raw = info.get("dividendYield) or info.get("trailingAnnualDividendYield")
        if dy_raw is not None:
            dy = dy_raw * 100 if dy_raw < 1 else dy_raw

    return {
        "ticker": ticker_b3,
        "preco_atual": preco_atual,
        "dy": dy if dy is not None else 0.0
    }

@st.cache_data(ttl=1800)
def buscar_fechamentos_14_pregoes(tickers_b3: tuple) -> dict:
    """Obtém os últimos 14 fechamentos para os ativos selecionados."""
    if not tickers_b3:
        return {}

    simbolos = [f"{ticker}.SA if not ticker.endswith(".SA") else ticker for ticker in tickers_b3]

    try:
        historico = yf.download(
            simbolos,
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return {}

    if historico.empty or "Close" not in historico:
        return {}

    fechamentos = historico["Close]
    if isinstance(fechamentos, pd.Series):
        fechamentos = fechamentos.to_frame(name=simbolos[0])

    resultado = {}
    for ticker, simbolo in zip(tickers_b3, simbolos):
        if simbolo in fechamentos.columns:
            serie = fechamentos[simbolo].dropna().tail(14)
            if not serie.empty:
                resultado[ticker] = serie

    return resultado

def exibir_painel_categoria(df_painel: pd.DataFrame, titulo: str, teto_vermelho: float):
    """Exibe métricas, gráficos e tabela de uma categoria de ativos."""
    if df_painel.empty:
        st.info(f"Não há ativos classificados como {titulo.lower()}.")
        return

    # Adicionando a coluna 'Trend' logo após o Ticker
    df_painel['Trend'] = df_painel['Diferença vs. CDI (R df_painel.sort_values(by="DY 12m (%)", ascending=False).reset_index(drop=True)
    # ... resto do código ...
    patrimonio = df_painel[Valor Atualizado (R Investido (R
    rentabilidade = ((patrimonio / custo_total) - 1) * 100 if custo_total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patrimônio Atual", f"R Investido", f"Rízo Total", f"R    c4.metric("Rentabilidade", f"{rentabilidade:+.2f}%")

    fig_bar = px.bar(
        df_painel.sort_values(by="Lucro/Prejuízo (R/Prejuízo (R title=f"<b>Lucro / Prejuízo Acumulado — {titulo}</b>",
        color="Lucro/Prejuízo (R    st.plotly_chart(fig_bar, use_container_width=True, key=f"resultado-{titulo}")

    st.subheader(f"Tendência dos Principais {titulo} — Últimos 14 Pregões")
    principais_ativos = df_painel.nlargest(8, "Valor Atualizado (R14_pregoes(tuple(principais_ativos))
    grade_graficos = st.columns(2)

    for indice, ticker in enumerate(principais_ativos):
        with grade_graficos[indice % 2]:
            fechamento = historicos.get(ticker)
            if fechamento is None or len(fechamento) < 2:
                st.warning(f"Histórico insuficiente para {ticker}.")
                continue

            variacao = ((fechamento.iloc[-1] / fechamento.iloc[0]) - 1) * 100
            df_grafico = fechamento.rename("Fechamento (Rchamento (R df_grafico,
                x="Data,
                y="Fechamento (R            )
            fig_tendencia.update_traces(line=dict(color="#2563EB", width=2))
            fig_tendencia.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=45, b=10),
                xaxis_title=None,
                yaxis_title="Preço (R            st.plotly_chart(
                fig_tendencia,
                use_container_width=True,
                key=f"tendencia-{titulo}-{ticker}",
            )

    st.subheader(f"Tabela Detalhada de {titulo} (Ordenada por DY Decrescente)")
    st.markdown(
        f"*Os ativos selecionados pelos piores DYs até o teto de **R em **vermelho** nas colunas **Ticker** e **DY 12m (%)**.*"
    )

    colunas_exibicao = [
        "Ticker, "Trend", "DY 12m (%)", "Valor Atualizado (R$)", "Lucro/Prejuízo (R$)",
        "Rentabilidade (%)", "Diferença vs. CDI (Risição", "Quantidade", "Preço Médio (R$)",
        "Preço Atual (R$)", "Custo Total Investido (Rabela = df_painel[colunas_exibicao].copy()

    def aplicar_estilo_pontual(data_frame_exibicao):
        estilos = pd.DataFrame("", index=data_frame_exibicao.index, columns=data_frame_exibicao.columns)
        
        # Estilos baseados no teto vermelho
        css_vermelho_forte = "color: white; background-color: #ef4444; font-weight: bold;"
        
        # Perfumaria CDI: Verde para positivo, Vermelho para negativo
        css_verde_cdi = "color: #15803d; background-color: #dcfce7; font-weight: bold;"
        css_vermelho_cdi = "color: #b91c1c; background-color: #fee2e2; font-weight: bold;"

        for idx, row in df_painel.iterrows():
            # Destaque de Teto Vermelho (Ticker e DY)
            if row["E_Vermelho"]:
                estilos.loc[idx, Ticker"] = css_vermelho_forte
                estilos.loc[idx, "DY 12m (%)"] = css_vermelho_forte
            
            # Perfumaria CDI (Setas e Performance)
            # COMENTÁRIO: Aqui começa a lógica de cores Verde/Vermelha para o CDI
            if row["Diferença vs. CDI (R[idx, "Trend"] = "color: #15803d; font-weight: bold;"
                estilos.loc[idx, "Diferença vs. CDI (R CDI (%)"] = css_verde_cdi
            else:
                estilos.loc[idx, "Trend"] = "color: #b91c1c; font-weight: bold;"
                estilos.loc[idx, "Diferença vs. CDI (R CDI (%)"] = css_vermelho_cdi
        
        return estilos

    df_estilizado = (
        df_tabela.style
        .apply(lambda _: aplicar_estilo_pontual(df_tabela), axis=None)
        .format({
            DY 12m (%)": "{:.2f}%",
            "Valor Atualizado (R$)": "R$ {:,.2f}",
            "Lucro/Prejuízo (R$)": "R$ {:,.2f}",
            "Rentabilidade (%)": "{:+.2f}%",
            "Diferença vs. CDI (R$)": "R$ {:+,.2f}",
            Acima/Abaixo do CDI (%)": "{:+.2f}%",
            "Quantidade": "{:,.0f}",
            "Preço Médio (R$)": "R$ {:,.2f}",
            "Preço Atual (R$)": "R$ {:,.2f}",
            "Custo Total Investido (R$)": "R$ {:,.2f}",
            "Soma Acumulada (R$)": "R$ {:,.2f}"
        }, na_rep="-")
    )
    st.dataframe(df_estilizado, use_container_width=True, height=500)

# INTERFACE DO STREAMLIT
st.title("📊 Dashboard da Carteira B3")
st.markdown("Análise consolidada dos extratos oficiais de Movimentações e Negociações da B3.")

st.sidebar.header("📂 Importação de Dados")
arquivo_upload = st.sidebar.file_uploader(
    "Envie a planilha `.xlsx` da B3 (Movimentações ou Negociações)",
    type=["xlsx"]
)

# Configuração dos tetos de destaque na Sidebar
st.sidebar.header("🔴 Tetos para Destaque em Vermelho")
teto_vermelho_acoes = st.sidebar.number_input(
    "Teto para Ações (R=0.0,
    step=1000.0,
    format="%.2f"
)
teto_vermelho_fiis = st.sidebar.number_input(
    "Teto para FIIs (R0,
    format="%.2f"
)

if arquivo_upload is not None:
    ativos = processar_movimentacoes_b3(arquivo_upload)

    if ativos.empty:
        st.error(f"Nenhum ativo com quantidade superior a {int(QTD_MINIMA_TOLERANCIA)} cotas/ações foi identificado.")
    else:
        st.sidebar.success(f"{len(ativos)} ativos com quantidade > {int(QTD_MINIMA_TOLERANCIA)} consolidados!")

        with st.spinner("Buscando cotações atualizadas na B3..."):
            tickers_list = ativos["ticker"].tolist()
            with ThreadPoolExecutor(max_workers=10) as executor:
                resultados = list(executor.map(buscar_dados_yfinance, tickers_list))

        mapa_dados = {res["ticker]: res for res in resultados}

        dados_completos = []
        for idx, row in ativos.iterrows():
            ticker = row["ticker"]
            dados = mapa_dados.get(ticker, {})

            qtd = row["quantidade]
            pm = row["preco_medio]
            custo_total = row[custo_total"]
            data_acq = row["data_aquisicao"]

            preco_atual = dados.get("preco_atual") or pm
            val_atualizado = qtd * preco_atual

            # CÁLCULO DE LUCRO/PREJUÍZO REFERENTE ÀS DATAS DE COMPRA
            lucro_prejuizo = val_atualizado - custo_total
            rentabilidade_pct = ((preco_atual / pm) - 1) * 100 if pm > 0 else 0
            valor_equivalente_cdi = row["valor_equivalente_cdi"]
            diferenca_cdi = (
                val_atualizado - valor_equivalente_cdi
                if pd.notna(valor_equivalente_cdi) else np.nan
            )
            rentabilidade_vs_cdi = (
                ((val_atualizado / valor_equivalente_cdi) - 1) * 100
                if pd.notna(valor_equivalente_cdi) and valor_equivalente_cdi > 0 else np.nan
            )

            dados_completos.append({
                Ticker": ticker,
                Tipo": row["tipo_ativo"],
                "DY 12m (%)": dados.get("dy", 0.0),
                "Valor Atualizado (Rjuizo,
                "Rentabilidade (%)": rentabilidade_pct,
                "Diferença vs. CDI (R do CDI (%)": rentabilidade_vs_cdi,
                "Data 1ª Aquisição: data_acq,
                "Quantidade: qtd,
                "Preço Médio (R "Custo Total Investido (Rivalente_cdi": row["valor_equivalente_cdi"] # Necessário para cálculos posteriores
            })

        df_base = pd.DataFrame(dados_completos)

        # Filtro final de segurança sobre valor atualizado
        df_base = df_base[df_base["Valor Atualizado (R
