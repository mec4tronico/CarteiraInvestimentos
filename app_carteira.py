"""app_carteira.py

Dashboard Interativo da Carteira de Investimentos
Ajustado para o Extrato Oficial de Movimentações da B3 com regras de exceção e tolerância de custódia.
"""

from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
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
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONFIGURAÇÕES E EXCEÇÕES DE REGRAS DE NEGÓCIO
# ==============================================================================
TICKER_MAP = {
    "ELET3": "AXIA3",
    "ELET6": "AXIA6",
    # Adicione novos de-para aqui caso outros ativos mudem de código
}

# Tolerância Mínima: Ativos com quantidade <= QTD_MINIMA_TOLERANCIA serão descartados
QTD_MINIMA_TOLERANCIA = 5.0


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


@st.cache_data(ttl=1800)
def processar_movimentacoes_b3(file_bytes) -> pd.DataFrame:
    """Processa o Extrato de Movimentações da B3, auditando e aplicando a
    tolerância mínima de quantidade (> 5 cotas/ações).
    """
    df = pd.read_excel(file_bytes)

    # Ordena cronologicamente (da movimentação mais antiga para a mais recente)
    df['Data_dt'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values(by='Data_dt', ascending=True)

    df['Quantidade_num'] = df['Quantidade'].apply(limpar_valor_numerico)
    df['Valor_num'] = df['Valor da Operação'].apply(limpar_valor_numerico)
    df['Preco_num'] = df['Preço unitário'].apply(limpar_valor_numerico)

    # Extrai o Ticker do produto
    df['Ticker'] = df['Produto'].astype(str).str.split('-').str[0].str.strip()

    # Mapeamento e substituição de tickers antigos (Ex: ELET3 -> AXIA3)
    df['Ticker'] = df['Ticker'].replace(TICKER_MAP)

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
        qtd = row['Quantidade_num']
        valor_op = row['Valor_num']
        data_op = row['Data_dt']

        # Filtra ativos inválidos ou títulos de Renda Fixa/Caixa
        if pd.isna(ticker) or ticker == "" or any(rf in ticker for rf in ['CDB', 'CFA', 'RDB', 'Tesouro']):
            continue

        if ticker not in posicoes:
            posicoes[ticker] = {
                'ticker': ticker,
                'quantidade': 0.0,
                'custo_total': 0.0,
                'primeira_compra': data_op,
            }

        # Identifica se é Entrada/Compra com valor financeiro
        is_compra = (mov in eventos_compra and tipo == 'Credito')
        # Identifica se é Saída/Venda
        is_venda = (mov == 'Venda' or (mov == 'Transferência - Liquidação' and tipo == 'Debito'))

        if is_compra:
            if valor_op == 0 and row['Preco_num'] > 0:
                valor_op = row['Preco_num'] * qtd

            posicoes[ticker]['quantidade'] += qtd
            posicoes[ticker]['custo_total'] += valor_op

            if posicoes[ticker]['primeira_compra'] is None or (pd.notnull(data_op) and data_op < posicoes[ticker]['primeira_compra']):
                posicoes[ticker]['primeira_compra'] = data_op

        elif is_venda and posicoes[ticker]['quantidade'] > 0:
            pm = posicoes[ticker]['custo_total'] / posicoes[ticker]['quantidade']
            posicoes[ticker]['quantidade'] = max(0.0, posicoes[ticker]['quantidade'] - qtd)
            posicoes[ticker]['custo_total'] = posicoes[ticker]['quantidade'] * pm

    # ==============================================================================
    # AUDITORIA DE SALDO: TOLERÂNCIA DE QUANTIDADE > 5
    # ==============================================================================
    resumo = []
    for t, p in posicoes.items():
        # Filtra estritamente ativos onde a quantidade é MAIOR que 5
        if p['quantidade'] > QTD_MINIMA_TOLERANCIA and p['custo_total'] > 0:
            pm = p['custo_total'] / p['quantidade']
            data_exibicao = p['primeira_compra'].strftime('%d/%m/%Y') if pd.notnull(p['primeira_compra']) else '-'
            resumo.append({
                'ticker': p['ticker'],
                'quantidade': p['quantidade'],
                'preco_medio': pm,
                'custo_total': p['custo_total'],
                'data_aquisicao': data_exibicao
            })

    return pd.DataFrame(resumo)


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

    preco_atual = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
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
        dy_raw = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        if dy_raw is not None:
            dy = dy_raw * 100 if dy_raw < 1 else dy_raw

    return {
        "ticker": ticker_b3,
        "preco_atual": preco_atual,
        "dy": dy if dy is not None else 0.0
    }


# INTERFACE DO STREAMLIT
st.title("📊 Dashboard da Carteira B3")
st.markdown("Análise consolidada do Extrato Oficial de Movimentações da B3.")

st.sidebar.header("📂 Importação de Dados")
arquivo_upload = st.sidebar.file_uploader("Envie a planilha `.xlsx` da B3", type=["xlsx"])

# Configuração da Entrada do APTO na Sidebar
st.sidebar.header("🏠 Entrada do APTO")
valor_entrada = st.sidebar.number_input(
    "Valor da Entrada (R$):",
    value=85000.0,
    step=1000.0,
    format="%.2f"
)

if arquivo_upload is not None:
    # Retorna apenas ativos com quantidade > 5
    ativos = processar_movimentacoes_b3(arquivo_upload)

    if ativos.empty:
        st.error(f"Nenhum ativo com quantidade superior a {int(QTD_MINIMA_TOLERANCIA)} cotas/ações foi identificado.")
    else:
        st.sidebar.success(f"{len(ativos)} ativos com quantidade > {int(QTD_MINIMA_TOLERANCIA)} consolidados!")

        with st.spinner("Buscando cotações atualizadas na B3..."):
            tickers_list = ativos["ticker"].tolist()
            with ThreadPoolExecutor(max_workers=10) as executor:
                resultados = list(executor.map(buscar_dados_yfinance, tickers_list))

        mapa_dados = {res["ticker"]: res for res in resultados}

        dados_completos = []
        for idx, row in ativos.iterrows():
            ticker = row["ticker"]
            dados = mapa_dados.get(ticker, {})

            qtd = row["quantidade"]
            pm = row["preco_medio"]
            custo_total = row["custo_total"]
            data_acq = row["data_aquisicao"]

            preco_atual = dados.get("preco_atual") or pm
            val_atualizado = qtd * preco_atual

            # CÁLCULO DE LUCRO/PREJUÍZO REFERENTE ÀS DATAS DE COMPRA
            lucro_prejuizo = val_atualizado - custo_total
            rentabilidade_pct = ((preco_atual / pm) - 1) * 100 if pm > 0 else 0

            dados_completos.append({
                "Ticker": ticker,
                "Data 1ª Aquisição": data_acq,
                "Quantidade": qtd,
                "Preço Médio (R$)": pm,
                "Preço Atual (R$)": preco_atual,
                "Custo Total Investido (R$)": custo_total,
                "Valor Atualizado (R$)": val_atualizado,
                "Lucro/Prejuízo (R$)": lucro_prejuizo,
                "Rentabilidade (%)": rentabilidade_pct,
                "DY 12m (%)": dados.get("dy", 0.0)
            })

        df_base = pd.DataFrame(dados_completos)

        # Filtro final de segurança sobre valor atualizado
        df_base = df_base[df_base["Valor Atualizado (R$)"] > 0].reset_index(drop=True)

        if df_base.empty:
            st.warning("Todos os ativos identificados possuem valor atual zerado.")
        else:
            # SELEÇÃO: Ordena por DY CRESCENTE -> Soma acumulada até o teto
            df_crescente = df_base.sort_values(by="DY 12m (%)", ascending=True).reset_index(drop=True)
            
            df_crescente["Soma Acumulada (R$)"] = df_crescente["Valor Atualizado (R$)"].cumsum()
            df_crescente["Soma Acumulada Anterior"] = df_crescente["Soma Acumulada (R$)"].shift(1, fill_value=0.0)

            df_crescente["E_Vermelho"] = df_crescente["Soma Acumulada Anterior"] < valor_entrada

            # EXIBIÇÃO: Reordena por DY DECRESCENTE
            df_final = df_crescente.sort_values(by="DY 12m (%)", ascending=False).reset_index(drop=True)

            # RESUMO EXECUTIVO (CARDS DE METRICAS)
            patrimonio_total = df_final["Valor Atualizado (R$)"].sum()
            custo_total_carteira = df_final["Custo Total Investido (R$)"].sum()
            lucro_total = patrimonio_total - custo_total_carteira
            rentabilidade_geral = ((patrimonio_total / custo_total_carteira) - 1) * 100 if custo_total_carteira > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Patrimônio Atual", f"R$ {patrimonio_total:,.2f}")
            c2.metric("Custo Total Investido", f"R$ {custo_total_carteira:,.2f}")
            c3.metric("Lucro / Prejuízo Total", f"R$ {lucro_total:,.2f}", delta=f"{lucro_total:,.2f}")
            c4.metric("Rentabilidade da Carteira", f"{rentabilidade_geral:+.2f}%")

            st.markdown("---")

            # GRÁFICOS
            g1, g2 = st.columns(2)

            with g1:
                fig_pie = px.pie(
                    df_final, 
                    values="Valor Atualizado (R$)", 
                    names="Ticker", 
                    title="<b>Alocação por Ativo</b>",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with g2:
                fig_bar = px.bar(
                    df_final.sort_values(by="Lucro/Prejuízo (R$)", ascending=True), 
                    x="Lucro/Prejuízo (R$)", 
                    y="Ticker", 
                    orientation="h",
                    title="<b>Lucro / Prejuízo Acumulado por Ativo (R$)</b>",
                    color="Lucro/Prejuízo (R$)",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # TABELA DETALHADA COM HIGHLIGHT PONTUAL
            st.subheader("📋 Tabela Detalhada de Posições (Ordenada por DY Decrescente)")
            st.markdown(
                f"*Os ativos selecionados pelos piores DYs para cobrir a entrada de **R$ {valor_entrada:,.2f}** "
                f"estão destacados em **vermelho** nas colunas **Ticker** e **DY 12m (%)** ao final da tabela.*"
            )

            colunas_exibicao = [
                "Ticker", 
                "Data 1ª Aquisição", 
                "Quantidade", 
                "Preço Médio (R$)", 
                "Preço Atual (R$)", 
                "Custo Total Investido (R$)", 
                "Valor Atualizado (R$)", 
                "Lucro/Prejuízo (R$)", 
                "Rentabilidade (%)", 
                "DY 12m (%)", 
                "Soma Acumulada (R$)"
            ]

            df_tabela = df_final[colunas_exibicao].copy()

            def aplicar_estilo_pontual(data_frame_exibicao):
                estilos = pd.DataFrame("", index=data_frame_exibicao.index, columns=data_frame_exibicao.columns)
                css_vermelho = "color: #991B1B; background-color: #FEE2E2; font-weight: bold;"
                
                for idx, row in df_final.iterrows():
                    if row["E_Vermelho"]:
                        if "Ticker" in estilos.columns:
                            estilos.loc[idx, "Ticker"] = css_vermelho
                        if "DY 12m (%)" in estilos.columns:
                            estilos.loc[idx, "DY 12m (%)"] = css_vermelho

                return estilos

            df_estilizado = (
                df_tabela
                .style
                .apply(lambda _: aplicar_estilo_pontual(df_tabela), axis=None)
                .format({
                    "Quantidade": "{:,.0f}",
                    "Preço Médio (R$)": "R$ {:,.2f}",
                    "Preço Atual (R$)": "R$ {:,.2f}",
                    "Custo Total Investido (R$)": "R$ {:,.2f}",
                    "Valor Atualizado (R$)": "R$ {:,.2f}",
                    "Lucro/Prejuízo (R$)": "R$ {:,.2f}",
                    "Rentabilidade (%)": "{:+.2f}%",
                    "DY 12m (%)": "{:.2f}%",
                    "Soma Acumulada (R$)": "R$ {:,.2f}"
                }, na_rep="-")
            )

            st.dataframe(
                df_estilizado,
                use_container_width=True,
                height=500
            )

else:
    st.info("👆 Por favor, envie a planilha de Movimentações (`.xlsx`) da B3 na barra lateral para carregar a análise.")
