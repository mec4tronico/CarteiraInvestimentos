import sqlite3
import pandas as pd
import os
from datetime import datetime
import numpy as np

DB_PATH = os.path.join('dados', 'carteira.db')

def conectar_banco():
    os.makedirs('dados', exist_ok=True)
    return sqlite3.connect(DB_PATH)

def inicializar_banco():
    """Cria as tabelas se não existirem."""
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
    conn.close()

def salvar_operacoes(df_novas):
    """Insere operações evitando duplicatas via hash único."""
    conn = conectar_banco()
    inseridos = 0
    for _, row in df_novas.iterrows():
        # Gerar hash único para evitar importar a mesma nota da B3 duas vezes
        h = f"{row['Data']}_{row['Ticker']}_{row['Quantidade_num']}_{row['Valor_num']}"
        try:
            conn.execute('''
                INSERT INTO operacoes (data, ticker, tipo_ativo, movimentacao, entrada_saida, quantidade, preco_unitario, valor_total, hash_operacao)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['Data_dt'].strftime('%Y-%m-%d'), row['Ticker'], row['tipo_ativo'], 
                  row['Movimentação'], row['Entrada/Saída'], row['Quantidade_num'], 
                  row['Preco_num'], row['Valor_num'], h))
            inseridos += 1
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    return inseridos

def listar_carteira():
    conn = conectar_banco()
    df = pd.read_sql_query("SELECT * FROM carteira", conn)
    conn.close()
    return df

def listar_operacoes():
    conn = conectar_banco()
    df = pd.read_sql_query("SELECT * FROM operacoes ORDER BY data ASC", conn)
    conn.close()
    return df

def gravar_posicao_final(df_resumo):
    conn = conectar_banco()
    conn.execute("DELETE FROM carteira")
    df_resumo.to_sql('carteira', conn, if_exists='append', index=False)
    conn.execute("INSERT OR REPLACE INTO configuracoes VALUES ('ultima_atualizacao', ?)", (datetime.now().strftime('%d/%m/%Y %H:%M'),))
    conn.commit()
    conn.close()

inicializar_banco()
