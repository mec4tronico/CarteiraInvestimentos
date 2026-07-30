"""
app_carteira.py

Dashboard Interativo da Carteira de Investimentos usando Streamlit,
yfinance e Plotly.
"""

import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Configuração da Página no Streamlit
st.set_page_config(
    page_title="Minha Carteira de Investimentos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nomes padrão das colunas da planilha da B3
COLUNA_TICKER = "Código de Negociação"
COLUNA_QUANTIDADE = "Quantidade"
COLUNA_VALOR = "Valor Atualizado"


@st.cache_data(ttl=3600)
def carregar_ativos_do_extrato(file_bytes) -> pd.DataFrame:
    """Lê todas as abas da planilha da B3 e consolida os ativos."""
    xls = pd.ExcelFile(file_bytes)
    partes = []

    for nome_aba in xls.sheet_names:
        df = xls.parse(nome_aba)
        if COLUNA_TICKER in df.columns:
            df = df.dropna(subset=[COLUNA_TICKER]).copy()
            partes.append(df)

    if not partes:
        return pd.DataFrame()

    todos = pd.concat(partes, ignore_index=True, sort=False)
    agrupado = (
        todos.groupby(COLUNA_TICKER, as_index=False)
        .agg({COLUNA_QUANTIDADE: "sum", COLUNA_VALOR: "sum"})
        .rename(columns={
            COLUNA_TICKER: "ticker",
            COLUNA_QUANTIDADE: "quantidade",
            COLUNA_VALOR: "valor_extrato_b3",
        })
    )
    return agrupado


def calcular_var_dias(hist: pd.DataFrame, dias: int):
    """Calcula a variação percentual e em R$ com base no histórico do yfinance."""
    if hist.empty or len(hist) < 2:
        return None, None

    preco_atual = float(hist["Close"].iloc[-1])
    data_atual = hist.index[-1]
    data_alvo = data_atual - pd.Timedelta(days=dias)

    historico_filtrado = hist[hist.index <= data_alvo]
    if historico_filtrado.empty:
        return None, None

    preco_base = float(historico_filtrado["Close"].iloc[-1])
    if preco_base == 0:
        return None, None

    var_pct = ((preco_atual / preco_base) - 1) * 100
    var_rs = preco_atual - preco_base
    return var_pct, var_rs


def buscar_dados_yfinance(ticker_b3: str) -> dict:
    """Busca dados de cotação atual, DY de 12 meses e variações no yfinance."""
    symbol = f"{ticker_b3}.SA" if not ticker_b3.endswith(".SA") else ticker_b3
    yticker = yf.Ticker(symbol)

    try:
        info = yticker.info or {}
    except Exception:
        info = {}

    try:
        hist = yticker.history(period="1y")
    except Exception:
        hist = pd.DataFrame()

    # Preço Atual
    preco_atual = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
    if preco_atual is None and not hist.empty:
        preco_atual = float(hist["Close"].iloc[-1])

    # CÁLCULO PRECISO DO DIVIDEND YIELD (Últimos 12 meses)
    dy = None
    try:
        dividends = yticker.dividends
        if dividends is not None and not dividends.empty and preco_atual and preco_atual > 0:
            uma_ano_atras = pd.Timestamp.now(tz=dividends.index.tz) - pd.Timedelta(days=365)
            divs_12m = dividends[dividends.index >= uma_ano_atras].sum()
            dy = (divs_12m / preco_atual) * 100
    except Exception:
        dy = None

    # Fallback caso não encontre no histórico
    if dy is None or dy == 0:
        dy_raw = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        if dy_raw is not None:
            dy = dy_raw * 100 if dy_raw < 1 else dy_raw

    # Variações
    var_30d_pct, var_30d_rs = calcular_var_dias(hist, 30)
    var_60d_pct, var_60d_rs = calcular_var_dias(hist, 60)
    var_90d_pct, var_90d_rs = calcular_var_dias(hist, 90)

    return {
        "preco_atual": preco_atual,
        "dy": dy,
        "var_30d_pct": var_30d_pct,
        "var_30d_rs": var_30d_rs,
        "var_60d_pct": var_60d_pct,
        "var_60d_rs": var_60d_rs,
        "var_90d_pct": var_90d_pct,
        "var_90d_rs": var_90d_rs,
    }


# INTERFACE DO APP STREAMLIT
st.title("📈 Dashboard da Carteira de Investimentos")
st.markdown("Consolidação automática de Ações e FIIs com cotações, variações e dividendos em tempo real.")

# Barra Lateral (Sidebar)
st.sidebar.header("⚙️ Configurações")
arquivo_upload = st.sidebar.file_uploader("Envie a planilha da B3 (.xlsx)", type=["xlsx"])

if arquivo_upload is not None:
    ativos = carregar_ativos_do_extrato(arquivo_upload)

    if ativos.empty:
        st.error("Não foi possível encontrar dados válidos no arquivo enviado.")
    else:
        st.sidebar.success(f"{len(ativos)} ativos encontrados!")

        progress_bar = st.progress(0)
        status_text = st.empty()

        dados_completos = []
        total_ativos = len(ativos)

        for idx, row in ativos.iterrows():
            ticker = row["ticker"]
            status_text.text(f"Atualizando cotações: {ticker} ({idx+1}/{total_ativos})")
            
            dados = buscar_dados_yfinance(ticker)
            
            qtd = row["quantidade"]
            preco = dados["preco_atual"] or 0
            val_extrato = row["valor_extrato_b3"]
            val_atualizado = qtd * preco if preco > 0 else val_extrato
            lucro_prejuizo = val_atualizado - val_extrato

            # Variação total na posição do investidor (Preço un. * Qtd)
            var_30d_tot_rs = (dados["var_30d_rs"] * qtd) if dados["var_30d_rs"] is not None else None
            var_60d_tot_rs = (dados["var_60d_rs"] * qtd) if dados["var_60d_rs"] is not None else None
            var_90d_tot_rs = (dados["var_90d_rs"] * qtd) if dados["var_90d_rs"] is not None else None

            dados_completos.append({
                "Ticker": ticker,
                "Quantidade": qtd,
                "Preço Atual (R$)": preco,
                "Valor Total (R$)": val_atualizado,
                "Lucro/Prejuízo (R$)": lucro_prejuizo,
                "DY 12m (%)": dados["dy"],
                "Var. 30d (%)": dados["var_30d_pct"],
                "Var. 30d (R$)": var_30d_tot_rs,
                "Var. 60d (%)": dados["var_60d_pct"],
                "Var. 60d (R$)": var_60d_tot_rs,
                "Var. 90d (%)": dados["var_90d_pct"],
                "Var. 90d (R$)": var_90d_tot_rs,
            })

            progress_bar.progress((idx + 1) / total_ativos)
            time.sleep(0.05)

        status_text.empty()
        progress_bar.empty()

        df_final = pd.DataFrame(dados_completos)

        # CARDS DE RESUMO (KPIs)
        patrimonio_total = df_final["Valor Total (R$)"].sum()
        lucro_total = df_final["Lucro/Prejuízo (R$)"].sum()
        dy_medio = df_final["DY 12m (%)"].dropna().mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Patrimônio Total", f"R$ {patrimonio_total:,.2f}")
        col2.metric("Lucro/Prejuízo Total", f"R$ {lucro_total:,.2f}", delta=f"{lucro_total:,.2f}")
        col3.metric("DY Médio da Carteira", f"{dy_medio:.2f}%" if pd.notnull(dy_medio) else "N/A")

        st.markdown("---")

        # GRÁFICOS INTERATIVOS
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            fig_pie = px.pie(
                df_final, 
                values="Valor Total (R$)", 
                names="Ticker", 
                title="<b>Alocação por Ativo</b>",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with g_col2:
            df_dy = df_final.dropna(subset=["DY 12m (%)"]).sort_values(by="DY 12m (%)", ascending=True)
            fig_bar = px.bar(
                df_dy, 
                x="DY 12m (%)", 
                y="Ticker", 
                orientation="h",
                title="<b>Ranking Dividend Yield 12m (%)</b>",
                text_auto=".2f"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # TABELA COMPLETA
        st.subheader("📋 Visão Detalhada da Carteira")

        st.dataframe(
            df_final.style.format({
                "Quantidade": "{:,.0f}",
                "Preço Atual (R$)": "R$ {:,.2f}",
                "Valor Total (R$)": "R$ {:,.2f}",
                "Lucro/Prejuízo (R$)": "R$ {:,.2f}",
                "DY 12m (%)": "{:.2f}%",
                "Var. 30d (%)": "{:+.2f}%",
                "Var. 30d (R$)": "R$ {:+,.2f}",
                "Var. 60d (%)": "{:+.2f}%",
                "Var. 60d (R$)": "R$ {:+,.2f}",
                "Var. 90d (%)": "{:+.2f}%",
                "Var. 90d (R$)": "R$ {:+,.2f}",
            }, na_rep="-"),
            use_container_width=True,
            height=450
        )

else:
    st.info("👆 Por favor, envie a planilha `.xlsx` da B3 na barra lateral para carregar seu dashboard.")