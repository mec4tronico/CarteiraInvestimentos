import sqlite3
import pandas as pd
import os
from datetime import datetime
import numpy as np
import json
from urllib.request import urlopen

DB_PATH = os.path.join('dados', 'carteira.db')


def conectar_banco():
    """Cria a pasta `dados/` (se necessário) e retorna uma conexão sqlite3.

    O chamador é responsável por fechar a conexão com `fechar_conexao`.
    """
    os.makedirs('dados', exist_ok=True)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    return conn


def fechar_conexao(conn: sqlite3.Connection):
    """Fecha a conexão passada (se existir)."""
    try:
        if conn:
            conn.close()
    except Exception:
        pass


def inicializar_banco():
    """Cria as tabelas necessárias se elas não existirem."""
    conn = conectar_banco()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS operacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data DATE,
            ticker TEXT,
            tipo_ativo TEXT,
            movimentacao TEXT,
            entrada_saida TEXT,
            quantidade REAL,
            preco_unitario REAL,
            valor_total REAL,
            numero_operacao TEXT,
            corretora TEXT,
            hash_operacao TEXT UNIQUE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS carteira (
            ticker TEXT PRIMARY KEY,
            tipo_ativo TEXT,
            quantidade REAL,
            preco_medio REAL,
            valor_investido REAL,
            data_primeira_compra DATE,
            valor_equivalente_cdi REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS configuracoes (
            chave TEXT PRIMARY KEY,
            valor TEXT
        )
    ''')
    conn.commit()
    fechar_conexao(conn)


# ---------------------- Funções de leitura/escrita ---------------------------

def listar_carteira() -> pd.DataFrame:
    """Retorna o conteúdo atual da tabela `carteira` como DataFrame."""
    conn = conectar_banco()
    try:
        df = pd.read_sql_query("SELECT * FROM carteira", conn, parse_dates=['data_primeira_compra'])
    finally:
        fechar_conexao(conn)
    return df


def listar_operacoes() -> pd.DataFrame:
    """Retorna todas as operações ordenadas por data (ascendente)."""
    conn = conectar_banco()
    try:
        df = pd.read_sql_query("SELECT * FROM operacoes ORDER BY data ASC", conn, parse_dates=['data'])
    finally:
        fechar_conexao(conn)
    return df


def salvar_operacoes(df_novas: pd.DataFrame) -> int:
    """Insere várias operações no banco evitando duplicatas pelo hash.

    Espera colunas mínimas em df_novas: Data_dt (datetime), Ticker, tipo_ativo,
    Movimentação, Entrada/Saída, Quantidade_num, Preco_num, Valor_num,
    numero_operacao (opcional), corretora (opcional).

    Retorna o número de operações efetivamente inseridas.
    """
    if df_novas is None or df_novas.empty:
        return 0

    conn = conectar_banco()
    inseridos = 0
    cursor = conn.cursor()

    for _, row in df_novas.iterrows():
        data_str = None
        try:
            data_str = row['Data_dt'].strftime('%Y-%m-%d')
        except Exception:
            # tentar outras colunas
            try:
                data_str = pd.to_datetime(row.get('Data')).strftime('%Y-%m-%d')
            except Exception:
                data_str = None

        ticker = str(row.get('Ticker', '')).strip().upper()
        tipo_ativo = row.get('tipo_ativo', '')
        movimentacao = row.get('Movimentação', '')
        entrada_saida = row.get('Entrada/Saída', '')
        quantidade = float(row.get('Quantidade_num', 0) or 0)
        preco_unitario = float(row.get('Preco_num', 0) or 0)
        valor_total = float(row.get('Valor_num', 0) or 0)
        numero_operacao = str(row.get('numero_operacao', ''))
        corretora = str(row.get('corretora', ''))

        # Gera hash simples que combina campos-chave para evitar duplicidades
        h = f"{data_str}_{ticker}_{quantidade}_{valor_total}_{numero_operacao}_{corretora}"

        try:
            cursor.execute('''
                INSERT INTO operacoes (data, ticker, tipo_ativo, movimentacao, entrada_saida, quantidade, preco_unitario, valor_total, numero_operacao, corretora, hash_operacao)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (data_str, ticker, tipo_ativo, movimentacao, entrada_saida, quantidade, preco_unitario, valor_total, numero_operacao, corretora, h))
            inseridos += 1
        except sqlite3.IntegrityError:
            # duplicata pela constraint UNIQUE no hash -> ignora
            continue
        except Exception:
            # ignora linhas problemáticas mas continua
            continue

    conn.commit()
    fechar_conexao(conn)
    return inseridos


# ---------------------- CDI / Cálculo de benchmark --------------------------

def buscar_cdi_diario(data_inicial: str, data_final: str) -> pd.Series:
    """Busca o CDI diário (série 12) no Banco Central do Brasil.

    Mantido aqui para que o cálculo do `valor_equivalente_cdi` possa ser
    realizado ao recalcular a carteira.
    """
    parametros = f"formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
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
        return pd.Series(dtype=float)


def calcular_valor_equivalente_cdi(aportes: list, cdi_diario: pd.Series) -> float:
    """Capitaliza cada aporte remanescente pelo CDI diário até a data atual."""
    if not aportes or cdi_diario.empty:
        return np.nan

    valor_equivalente = 0.0
    for aporte in aportes:
        taxas = cdi_diario[cdi_diario.index >= aporte['data']]
        fator = (1 + taxas / 100).prod() if not taxas.empty else 1.0
        valor_equivalente += aporte['valor'] * fator
    return valor_equivalente


# ---------------------- Recalcular / Atualizar carteira --------------------

def atualizar_carteira() -> int:
    """Recalcula a posição atual da carteira a partir de todas as operações.

    Aplica as operações em ordem cronológica, tratando compras e vendas e
    armazenando o resultado na tabela `carteira`.

    Retorna o número de tickers gravados na tabela `carteira`.
    """
    # Carregue todas as operações em ordem
    df_ops = listar_operacoes()
    if df_ops.empty:
        # limpa carteira
        conn = conectar_banco()
        conn.execute("DELETE FROM carteira")
        conn.commit()
        fechar_conexao(conn)
        return 0

    posicoes = {}

    for _, row in df_ops.iterrows():
        ticker = str(row['ticker']).upper()
        tipo = row.get('tipo_ativo', '')
        movimentacao = str(row.get('movimentacao', ''))
        entrada_saida = str(row.get('entrada_saida', ''))
        qtd = float(row.get('quantidade', 0) or 0)
        valor = float(row.get('valor_total', 0) or 0)
        data_op = pd.to_datetime(row.get('data'))

        if ticker == '' or pd.isna(qtd) or qtd == 0:
            continue

        if ticker not in posicoes:
            posicoes[ticker] = {
                'ticker': ticker,
                'quantidade': 0.0,
                'custo_total': 0.0,
                'primeira_compra': None,
                'tipo_ativo': tipo,
                'aportes': []
            }

        # Define regras simples de compra/venda com base em movimentacao/entrada_saida
        is_compra = (movimentacao.lower().startswith('comp') or entrada_saida.lower() == 'credito')
        is_venda = (movimentacao.lower().startswith('ven') or entrada_saida.lower() == 'debito')

        if is_compra:
            # Se valor estiver zerado mas houver preco_unitario, calcule
            preco_unit = float(row.get('preco_unitario', 0) or 0)
            if valor == 0 and preco_unit > 0:
                valor = preco_unit * qtd

            posicoes[ticker]['quantidade'] += qtd
            posicoes[ticker]['custo_total'] += valor
            posicoes[ticker]['aportes'].append({
                'data': data_op.normalize(),
                'quantidade': qtd,
                'valor': valor,
            })
            if posicoes[ticker]['primeira_compra'] is None:
                posicoes[ticker]['primeira_compra'] = data_op

        elif is_venda and posicoes[ticker]['quantidade'] > 0:
            quantidade_anterior = posicoes[ticker]['quantidade']
            if quantidade_anterior <= 0:
                continue
            pm = posicoes[ticker]['custo_total'] / quantidade_anterior if quantidade_anterior > 0 else 0
            quantidade_vendida = min(qtd, quantidade_anterior)
            fator_remanescente = 1 - (quantidade_vendida / quantidade_anterior)
            posicoes[ticker]['quantidade'] = max(0.0, quantidade_anterior - qtd)
            posicoes[ticker]['custo_total'] = posicoes[ticker]['quantidade'] * pm

            # Ajusta aportes proporcionalmente
            for aporte in posicoes[ticker]['aportes']:
                aporte['quantidade'] *= fator_remanescente
                aporte['valor'] *= fator_remanescente
            posicoes[ticker]['aportes'] = [a for a in posicoes[ticker]['aportes'] if a['quantidade'] > 0]

            if posicoes[ticker]['quantidade'] == 0:
                posicoes[ticker]['primeira_compra'] = None

    # Calcula CDI para aportes (se houver)
    todas_datas_aportes = [a['data'] for p in posicoes.values() for a in p['aportes']]
    if todas_datas_aportes:
        data_inicial = min(todas_datas_aportes).strftime('%d/%m/%Y')
        data_final = pd.Timestamp.today().strftime('%d/%m/%Y')
        cdi_diario = buscar_cdi_diario(data_inicial, data_final)
    else:
        cdi_diario = pd.Series(dtype=float)

    # Monta DataFrame resumo e grava na tabela carteira
    linhas = []
    for t, p in posicoes.items():
        qtd = p['quantidade']
        custo_total = p['custo_total']
        if qtd > 0 and custo_total > 0:
            pm = custo_total / qtd if qtd > 0 else 0
            valor_eq_cdi = calcular_valor_equivalente_cdi(p['aportes'], cdi_diario)
            data_primeira = p['primeira_compra']
            linhas.append({
                'ticker': t,
                'tipo_ativo': p.get('tipo_ativo', ''),
                'quantidade': qtd,
                'preco_medio': pm,
                'valor_investido': custo_total,
                'data_primeira_compra': data_primeira.strftime('%Y-%m-%d') if data_primeira is not None else None,
                'valor_equivalente_cdi': float(valor_eq_cdi) if pd.notna(valor_eq_cdi) else None,
            })

    df_resumo = pd.DataFrame(linhas)

    conn = conectar_banco()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM carteira")
    if not df_resumo.empty:
        df_resumo.to_sql('carteira', conn, if_exists='append', index=False)
    # atualiza configuração
    cursor.execute("INSERT OR REPLACE INTO configuracoes (chave, valor) VALUES ('ultima_atualizacao', ?)", (datetime.now().strftime('%d/%m/%Y %H:%M'),))
    conn.commit()
    fechar_conexao(conn)

    return len(df_resumo)


# Inicializa banco ao importar o módulo (comportamento antigo preservado)
inicializar_banco()
