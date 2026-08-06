"""app_carteira.py
Dashboard Interativo da Carteira de Investimentos (refatorado para usar SQLite)

Esta versão preserva a interface do Streamlit e altera apenas a origem
dos dados para utilizar o módulo database.py (dados/carteira.db).

Comentários importantes:
- A importação da planilha B3 continua disponível pelo uploader na sidebar.
- Ao enviar o arquivo, operações novas são inseridas em `operacoes` e a
  tabela `carteira` é recalculada com `recalcular_carteira()`.
- Se o banco estiver vazio, o app instrui o usuário a importar a planilha.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Importa a camada de dados centralizada
from database import (
    inicializar_banco,
    listar_carteira,
    listar_operacoes,
    salvar_operacoes,
    recalcular_carteira,
    buscar_cdi_diario,
    calcular_valor_equivalente_cdi,
)

# Configuração da Página no Streamlit
st.set_page_config(
    page_title="Minha Carteira B3 - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
TICKER_MAP = {
    "ELET3": "AXIA3",
    "ELET6": "AXIA6",
}

# Lista reduzida de FIIs (preservada do original para classificação)
TICKERS_FII_IFIX = {
    'AFHI11', 'ALZR11', 'AZPL11', 'BBIG11', 'BCRI11', 'BCIA11', 'BPML11',
    'BRCO11', 'BRCR11', 'BROF11', 'BTAL11', 'BTLG11', 'BTHF11', 'BTCI11',
}

QTD_MINIMA_TOLERANCIA = 5.0

# Garante inicialização do banco (cria dados/carteira.db e tabelas se necessário)
inicializar_banco()

# ==============================================================================
# UTILITÁRIOS (mantidos e corrigidos)
# ==============================================================================

def limpar_valor_numerico(val):
    """Converte valores no formato B3/Excel para float."""
    if pd.isna(val) or val == "-" or str(val).strip() == "":
        return 0.0
    try:
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).replace('.', '').replace(',', '.')
        return float(val_str)
    except Exception:
        return 0.0


def classificar_tipo_ativo(produto, ticker) -> str:
    """Classifica ativos. Se termina em 11 e não for FII, é Unit (Ação)."""
    descricao = str(produto).upper()
    ticker = str(ticker).upper().strip()

    if ticker in TICKERS_FII_IFIX or "FII" in descricao or "IMOBILIARIO" in descricao:
        return "FII"
    if ticker.endswith('11'):
        return "Unit"
    return "Ação"


def _ensure_series(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    """Garante que exista uma Series para a coluna (fallback com valores default)."""
    if col in df.columns:
        return df[col]
    # Se df estiver vazio, retorna Series vazia com mesmo index
    return pd.Series([default] * len(df), index=df.index)


# ==============================================================================
# Leitura e normalização da planilha B3
# ==============================================================================
@st.cache_data(ttl=3600)
def _ler_planilha_b3(file_bytes) -> pd.DataFrame:
    """Lê e normaliza a planilha B3 para o formato interno de operações.

    Produz as colunas esperadas por salvar_operacoes():
    - Data_dt (datetime), Ticker, tipo_ativo, Movimentação, Entrada/Saída,
      Quantidade_num, Preco_num, Valor_num, numero_operacao, corretora
    """
    try:
        df = pd.read_excel(file_bytes)
    except Exception as e:
        raise ValueError(f"Falha ao ler arquivo Excel: {e}")

    # Detecta se é o relatório "Negociações" (nomes mais longos)
    colunas_negociacoes = {
        'Data do Negócio', 'Tipo de Movimentação', 'Código de Negociação', 'Quantidade', 'Preço', 'Valor'
    }
    if colunas_negociacoes.issubset(set(df.columns)):
        df = df.rename(columns={
            'Data do Negócio': 'Data',
            'Tipo de Movimentação': 'Movimentação',
            'Código de Negociação': 'Ticker',
            'Preço': 'Preço unitário',
            'Valor': 'Valor da Operação',
        })
        # consolida sufixo de mercado fracionário (ex: PETR4F -> PETR4)
        df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper().str.replace(r'F$', '', regex=True)
        df['Entrada/Saída'] = df['Movimentação'].map({'Compra': 'Credito', 'Venda': 'Debito'}).fillna('')
        df['Produto'] = df['Ticker']
    else:
        # Espera o formato de Movimentações (Extrato)
        colunas_mov = {'Data', 'Movimentação', 'Entrada/Saída', 'Produto', 'Quantidade', 'Preço unitário', 'Valor da Operação'}
        faltantes = colunas_mov - set(df.columns)
        if faltantes:
            # Tentativa resiliente: tenta inferir colunas similares
            raise ValueError('Formato de planilha B3 não reconhecido. Colunas ausentes: ' + ', '.join(sorted(faltantes)))
        # Extrai ticker do campo Produto (formato "TICKER - ...")
        df['Ticker'] = df['Produto'].astype(str).str.split('-').str[0].str.strip().str.upper()

    # normalizações e mapeamentos defensivos (garantir Series para cada coluna)
    df['Data_dt'] = pd.to_datetime(_ensure_series(df, 'Data'), dayfirst=True, errors='coerce')

    # Quantidade e Valor
    df['Quantidade_num'] = _ensure_series(df, 'Quantidade').apply(limpar_valor_numerico)
    df['Valor_num'] = _ensure_series(df, 'Valor da Operação').apply(limpar_valor_numerico)

    # Preço unitário: preferir coluna explícita, senão calcular como Valor / Quantidade
    if 'Preço unitário' in df.columns:
        df['Preco_num'] = df['Preço unitário'].apply(limpar_valor_numerico)
    else:
        # evita divisão por zero
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Preco_num'] = (df['Valor_num'] / df['Quantidade_num'].replace(0, np.nan)).fillna(0.0)

    # Aplica map de tickers antigos
    df['Ticker'] = df['Ticker'].replace(TICKER_MAP)

    # Produto / tipo_ativo: usar Produto quando disponível senão fallback para Ticker
    prod_series = _ensure_series(df, 'Produto', default='')
    prod_series = prod_series.astype(str) if not prod_series.empty else prod_series
    df['tipo_ativo'] = prod_series.apply(lambda x: classificar_tipo_ativo(x, x)) if not prod_series.empty else df['Ticker'].apply(lambda x: classificar_tipo_ativo(x, x))

    # Movimentação e Entrada/Saída como strings
    mov = _ensure_series(df, 'Movimentação', default='')
    ent = _ensure_series(df, 'Entrada/Saída', default='')
    df['Movimentação'] = mov.astype(str)
    df['Entrada/Saída'] = ent.astype(str)

    # Colunas opcionais
    df['numero_operacao'] = _ensure_series(df, 'numero_operacao', default='').astype(str)
    df['corretora'] = _ensure_series(df, 'corretora', default='').astype(str)

    # Ordena cronologicamente
    df = df.sort_values(by='Data_dt', ascending=True)

    # Seleciona apenas colunas que salvaremos
    cols_needed = ['Data_dt', 'Ticker', 'tipo_ativo', 'Movimentação', 'Entrada/Saída', 'Quantidade_num', 'Preco_num', 'Valor_num', 'numero_operacao', 'corretora']
    for c in cols_needed:
        if c not in df.columns:
            df[c] = pd.Series([None] * len(df), index=df.index)

    return df[cols_needed]


# ==============================================================================
# Integração do uploader com o banco
# ==============================================================================
def importar_e_atualizar(file_bytes):
    """Lê planilha, salva operações novas e recalcula carteira.

    Retorna (n_inseridas, n_tickers_atualizados)
    """
    try:
        df_ops = _ler_planilha_b3(file_bytes)
    except Exception as e:
        st.error(f"Erro ao ler a planilha: {e}")
        # log de colunas e tipos para ajudar debugging
        try:
            # tenta abrir com pandas para mostrar colunas se possível
            df_tmp = pd.read_excel(file_bytes)
            st.error(f"Colunas detectadas no arquivo: {list(df_tmp.columns)}")
        except Exception:
            pass
        return 0, 0

    # Insere operações (salvar_operacoes faz deduplicação por hash)
    try:
        n_inseridas = salvar_operacoes(df_ops)
    except Exception as e:
        st.error(f"Falha ao salvar operações no banco: {e}")
        n_inseridas = 0

    # Recalcula carteira
    try:
        n_tickers = recalcular_carteira()
    except Exception as e:
        st.error(f"Falha ao recalcular a carteira: {e}")
        n_tickers = 0

    return n_inseridas, n_tickers


# ==============================================================================
# Funções de dados / cotações (mantidas e corrigidas)
# ==============================================================================

def buscar_dados_yfinance(ticker_b3: str) -> dict:
    """Busca cotações atuais e métricas via Yahoo Finance para ativos da B3."""
    symbol = f"{ticker_b3}.SA" if not ticker_b3.endswith('.SA') else ticker_b3
    yticker = yf.Ticker(symbol)

    try:
        info = yticker.info or {}
    except Exception:
        info = {}

    try:
        hist = yticker.history(period="1mo")
    except Exception:
        hist = pd.DataFrame()

    preco_atual = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
    if (preco_atual is None or preco_atual == 0) and not hist.empty:
        try:
            preco_atual = float(hist['Close'].iloc[-1])
        except Exception:
            preco_atual = None

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
        dy_raw = info.get('dividendYield') or info.get('trailingAnnualDividendYield')
        if dy_raw is not None:
            dy = dy_raw * 100 if dy_raw < 1 else dy_raw

    return {
        'ticker': ticker_b3,
        'preco_atual': preco_atual,
        'dy': dy if dy is not None else 0.0,
        'hist': hist,
    }


@st.cache_data(ttl=1800)
def buscar_fechamentos_14_pregoes(tickers_b3: tuple) -> dict:
    """Obtém os últimos 14 fechamentos para os ativos selecionados."""
    if not tickers_b3:
        return {}

    simbolos = [f"{ticker}.SA" if not ticker.endswith('.SA') else ticker for ticker in tickers_b3]

    try:
        historico = yf.download(
            simbolos,
            period='1mo',
            interval='1d',
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return {}

    if historico.empty:
        return {}

    fechamentos = historico['Close'] if 'Close' in historico else historico

    resultado = {}
    for ticker, simbolo in zip(tickers_b3, simbolos):
        if simbolo in fechamentos.columns:
            serie = fechamentos[simbolo].dropna().tail(14)
            if not serie.empty:
                resultado[ticker] = serie
    return resultado


# ==============================================================================
# RENDERIZAÇÃO DA INTERFACE (mantida)
# ==============================================================================
st.title('📊 Dashboard da Carteira B3')
st.markdown('Análise consolidada dos extratos oficiais de Movimentações e Negociações da B3.')

st.sidebar.header('📂 Importação de Dados')
arquivo_upload = st.sidebar.file_uploader(
    'Envie a planilha `.xlsx` da B3 (Movimentações ou Negociações)',
    type=['xlsx']
)

# Tetos para destaque (mantidos)
st.sidebar.header('🔴 Tetos para Destaque em Vermelho')
teto_vermelho_acoes = st.sidebar.number_input('Teto para Ações (R$)', value=0.0, step=1000.0, format='%.2f')
teto_vermelho_fiis = st.sidebar.number_input('Teto para FIIs (R$)', value=0.0, format='%.2f')

# Se usuário enviar arquivo -> importar e recarregar
if arquivo_upload is not None:
    with st.spinner('Importando e atualizando a carteira...'):
        n_ins, n_ticks = importar_e_atualizar(arquivo_upload)
    if n_ins > 0:
        st.sidebar.success(f'{n_ins} operações novas importadas e carteira atualizada ({n_ticks} tickers).')
    else:
        st.sidebar.info('Nenhuma operação nova encontrada ou foram detectados apenas registros duplicados.')

# Leitura da carteira a partir do banco (ETAPA 2)
df_carteira = listar_carteira()

if df_carteira is None or df_carteira.empty:
    st.info('A tabela "carteira" está vazia. Use a opção "Importação de Dados" na barra lateral para enviar a planilha Negociações B3 e popular o banco.')
else:
    # Prepara lista de tickers
    tickers_list = df_carteira['ticker'].astype(str).str.strip().str.upper().tolist()

    # Busca cotações em paralelo
    with st.spinner('Buscando cotações atualizadas na B3...'):
        with ThreadPoolExecutor(max_workers=8) as executor:
            resultados = list(executor.map(buscar_dados_yfinance, tickers_list))

    mapa_dados = {res['ticker']: res for res in resultados}

    dados_completos = []
    for _, row in df_carteira.iterrows():
        ticker = row['ticker']
        dados = mapa_dados.get(ticker, {})

        qtd = float(row.get('quantidade') or 0)
        pm = float(row.get('preco_medio') or 0)
        custo_total = float(row.get('valor_investido') or 0)
        data_acq = row.get('data_primeira_compra')

        preco_atual = dados.get('preco_atual') or pm or 0
        val_atualizado = qtd * (preco_atual or 0)

        lucro_prejuizo = val_atualizado - custo_total
        rentabilidade_pct = ((preco_atual / pm) - 1) * 100 if pm > 0 else 0
        valor_equivalente_cdi = row.get('valor_equivalente_cdi')
        diferenca_cdi = val_atualizado - valor_equivalente_cdi if pd.notna(valor_equivalente_cdi) else np.nan
        rentabilidade_vs_cdi = ((val_atualizado / valor_equivalente_cdi) - 1) * 100 if pd.notna(valor_equivalente_cdi) and valor_equivalente_cdi > 0 else np.nan

        dados_completos.append({
            'Ticker': ticker,
            'Tipo': row.get('tipo_ativo', ''),
            'DY 12m (%)': dados.get('dy', 0.0),
            'Valor Atualizado (R$)': val_atualizado,
            'Lucro/Prejuízo (R$)': lucro_prejuizo,
            'Rentabilidade (%)': rentabilidade_pct,
            'Diferença vs. CDI (R$)': diferenca_cdi,
            'Rentabilidade vs CDI (%)': rentabilidade_vs_cdi,
            'Data 1ª Aquisição': data_acq,
            'Quantidade': qtd,
            'Preço Médio (R$)': pm,
            'Preço Atual (R$)': preco_atual,
            'Custo Total Investido (R$)': custo_total,
            'Valor Equivalente CDI (R$)': valor_equivalente_cdi,
        })

    df_base = pd.DataFrame(dados_completos)

    # Exibição de métricas resumidas
    st.subheader('Resumo da Carteira')
    total_patrimonio = df_base['Valor Atualizado (R$)'].sum()
    total_investido = df_base['Custo Total Investido (R$)'].sum()
    total_lucro = df_base['Lucro/Prejuízo (R$)'].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric('Patrimônio Atual', f'R$ {total_patrimonio:,.2f}')
    c2.metric('Total Investido', f'R$ {total_investido:,.2f}')
    c3.metric('Lucro / Prejuízo', f'R$ {total_lucro:,.2f}')

    # Gráfico: patrimônio por ativo (mantém visual semelhante ao original)
    st.subheader('Patrimônio por Ativo')
    fig = px.bar(df_base.sort_values(by='Valor Atualizado (R$)', ascending=False),
                 x='Ticker', y='Valor Atualizado (R$)', color='Tipo', title='<b>Patrimônio por Ativo</b>')
    st.plotly_chart(fig, use_container_width=True)

    # Tabela detalhada (ordenada por DY decrescente como no original)
    st.subheader('Tabela Detalhada')
    if 'DY 12m (%)' in df_base.columns:
        df_tabela = df_base.sort_values(by='DY 12m (%)', ascending=False).reset_index(drop=True)
    else:
        df_tabela = df_base.copy()

    # Formatação simples para exibição; não alteramos colunas disponíveis
    st.dataframe(df_tabela, use_container_width=True, height=600)

# fim do app
