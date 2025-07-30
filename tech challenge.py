import streamlit as st
import os
import pandas as pd
import numpy as np
import random
from datetime import datetime
#Criação de indivíduo com pesos aleatórios para ativos aleatórios
def criar_individuo(num_ativos: int, num_ativos_desejados: int):
    pesos_finais = np.zeros(num_ativos)
    indices_selecionados = np.random.choice(
        a=num_ativos,
        size=num_ativos_desejados,
        replace=False
    )
    pesos_distribuidos = np.random.dirichlet(np.ones(num_ativos_desejados))
    pesos_finais[indices_selecionados] = pesos_distribuidos
    return pesos_finais
# Criação da população inicial com indivíduos aleatórios
def criar_populacao_inicial(tamanho_populacao: int, num_ativos: int, num_ativos_desejados: int):
    return [criar_individuo(num_ativos, num_ativos_desejados) for _ in range(tamanho_populacao)]
#Calcular o fitness de um indivíduo
def calcular_fitness(individuo, ativos_df, risco_maximo, data_retirada, valor_total):
    retorno_carteira = np.sum((valor_total * individuo) * (1 + (ativos_df['rentabilidade_total'] / 100)))
    risco_carteira = np.sum(individuo * ativos_df['risco'])

    fitness = retorno_carteira

    if risco_carteira > risco_maximo:
        fitness = -1

    for i, peso in enumerate(individuo):
        if peso > 0: # Apenas para ativos que estão na carteira
            if ativos_df['vencimento'].iloc[i] > data_retirada:
                fitness = -1
            if (ativos_df['investimento_maximo'].iloc[i] < valor_total * peso):
                fitness = -1
            if (ativos_df['investimento_minimo'].iloc[i] > valor_total * peso):
                fitness = -1

    return fitness
def selecao(populacao, fitness_valores, num_selecionados):
    indices_ordenados = np.argsort(fitness_valores)[::-1]
    return [populacao[i] for i in indices_ordenados[:num_selecionados]]
def cruzamento(pai1, pai2, num_ativos_desejados):
    indices_pai1 = set(np.where(pai1 > 0)[0])
    indices_pai2 = set(np.where(pai2 > 0)[0])
    pool_indices_heranca = list(indices_pai1.union(indices_pai2))

    if len(pool_indices_heranca) < num_ativos_desejados:
        num_indices_faltantes = num_ativos_desejados - len(pool_indices_heranca)
        todos_indices_possiveis = set(range(len(pai1)))
        indices_disponiveis = list(todos_indices_possiveis - set(pool_indices_heranca))
        random.shuffle(indices_disponiveis)
        indices_novos = indices_disponiveis[:num_indices_faltantes]
        pool_indices_heranca.extend(indices_novos)

    random.shuffle(pool_indices_heranca)
    indices_herdados = pool_indices_heranca[:num_ativos_desejados]
    filho = np.zeros_like(pai1)

    for indice in indices_herdados:
        filho[indice] = pai1[indice] if pai1[indice] > pai2[indice] else pai2[indice]
        filho[indice] = 0.1 if filho[indice] == 0 else filho[indice]

    soma_pesos_filho = np.sum(filho)
    filho_normalizado = filho / soma_pesos_filho
    return filho_normalizado
def mutacao(individuo, taxa_mutacao):
    if random.random() < taxa_mutacao:
        indices_individuo = list(set(np.where(individuo > 0)[0]))
        random.shuffle(indices_individuo)
        indice_escolhido = indices_individuo[0]
        alteracao = random.uniform(-0.1, 0.1)
        individuo[indice_escolhido] += alteracao
        individuo[individuo < 0] = 0.01
        return individuo / np.sum(individuo)
    return individuo
def rodar_algoritmo_genetico(ativos_df, valor_total, risco_maximo, data_retirada, num_ativos_desejados, progress_bar, qtd_geracoes=200, tamanho_populacao=100, taxa_mutacao=0.1, num_elite=5):
    num_ativos = len(ativos_df)
    populacao = criar_populacao_inicial(tamanho_populacao, num_ativos, num_ativos_desejados)
    melhor_fitness_geral = 0
    melhor_carteira_geral = None

    status_text = st.empty()

    for geracao in range(qtd_geracoes):
        fitness_valores = [calcular_fitness(ind, ativos_df, risco_maximo, data_retirada, valor_total) for ind in populacao]
        melhor_fitness_geracao = max(fitness_valores)

        if melhor_fitness_geracao > melhor_fitness_geral:
            melhor_fitness_geral = melhor_fitness_geracao
            melhor_carteira_geral = populacao[np.argmax(fitness_valores)]

        status_text.text(f"Geração {geracao}/{qtd_geracoes} - Melhor Fitness: R$ {melhor_fitness_geral:,.2f}")
        progress_bar.progress((geracao + 1) / qtd_geracoes)

        nova_populacao = []
        elite = selecao(populacao, fitness_valores, num_elite)
        nova_populacao.extend(elite)

        while len(nova_populacao) < (tamanho_populacao * 0.7):
            pai1, pai2 = random.choices(elite, k=2)
            filho = cruzamento(pai1, pai2, num_ativos_desejados)
            filho = mutacao(filho, taxa_mutacao)
            nova_populacao.append(filho)

        while len(nova_populacao) < tamanho_populacao:
            nova_populacao.append(criar_individuo(num_ativos, num_ativos_desejados))

        populacao = nova_populacao

    return melhor_carteira_geral
def exibir_resultados(carteira, ativos_df, valor_total):
    if carteira is None:
        st.error("Não foi possível encontrar uma carteira que atenda a todos os critérios. Tente ajustar os parâmetros.")
        return

    carteira_final = ativos_df.copy()
    carteira_final['peso'] = carteira
    carteira_final['alocacao'] = carteira * valor_total
    carteira_final = carteira_final[carteira_final['peso'] > 0.001]

    retorno_final = np.sum(carteira * ativos_df['rentabilidade'])
    risco_final = np.sum(carteira * ativos_df['risco'])

    st.subheader("Resultados da Otimização")
    col1, col2 = st.columns(2)
    col1.metric("Retorno Estimado (a.a)", f"{retorno_final:.2f}%")
    col2.metric("Risco da Carteira", f"{risco_final:.2f}")

    st.subheader("Alocação por Ativo")
    carteira_final_formatada = carteira_final[['nome', 'peso', 'alocacao', 'rentabilidade', 'risco', 'vencimento']].copy()
    carteira_final_formatada['vencimento'] = carteira_final_formatada['vencimento'].dt.strftime('%d/%m/%Y')

    st.dataframe(carteira_final_formatada.style
                 .format({'peso': '{:.2%}', 'alocacao': 'R$ {:,.2f}', 'rentabilidade': '{:.2f}%'})
                 .hide(axis="index")
                 )
def main():
    st.set_page_config(layout="wide", page_title="Otimizador de Carteira de Investimentos")

    st.title("Carteira Recomendada com Algoritmo Genético")

    # --- Barra Lateral para Entradas do Usuário ---
    st.sidebar.header("Parâmetros do Investidor")

    VALOR_INVESTIMENTO_TOTAL = st.sidebar.number_input(
        "Valor Total do Investimento (R$)",
        min_value=1000.0,
        value=100000.0,
        step=1000.0,
        format="%.2f"
    )

    QUANTIDADE_ATIVOS = st.sidebar.number_input(
        "Quantidade de Ativos na Carteira",
        min_value=1,
        value=10,
        step=1
    )

    GERACOES = st.sidebar.number_input(
        "Quantidade de Gerações",
        min_value=1,
        max_value=1000,
        value=200,
        step=1
    )

    TAMANHO_POPULACAO = st.sidebar.number_input(
        "Tamanho da População",
        min_value=1,
        max_value=300,
        value=100,
        step=1
    )

    TAXA_MUTACAO = st.sidebar.number_input(
        "Taxa de Mutação (%)",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

    NUM_ELITE = st.sidebar.number_input(
        "Taxa de população de elite (%)",
        min_value=1,
        max_value=100,
        value=5,
        step=1
    )

    RISCO_ACEITAVEL = st.sidebar.slider(
        "Nível de Risco Aceitável (1 a 40)",
        min_value=1,
        max_value=40,
        value=5,
        step=1
    )

    DATA_RETIRADA_STR = st.sidebar.text_input(
        "Data de Retirada (dd/mm/aaaa)",
        value="01/01/2030"
    )

    if st.sidebar.button("Otimizar Carteira"):
        try:
            # --- Carregamento e Preparação dos Dados ---
            DATA_ATUAL = datetime.now()
            DATA_RETIRADA = datetime.strptime(DATA_RETIRADA_STR, "%d/%m/%Y")

            st.header("Resumo dos Parâmetros")
            st.write(f"**Valor a ser investido:** R$ {VALOR_INVESTIMENTO_TOTAL:,.2f}")
            st.write(f"**Risco máximo aceitável:** {RISCO_ACEITAVEL}")
            st.write(f"**Data de necessidade do dinheiro:** {DATA_RETIRADA_STR}")
            st.write("---")

            # Assumindo que 'ativos.csv' está no mesmo diretório
            caminho_do_csv = 'ativos.csv'
            if not os.path.exists(caminho_do_csv):
                st.error(f"Arquivo 'ativos.csv' não encontrado. Por favor, coloque o arquivo no mesmo diretório do script.")
                return

            df = pd.read_csv(caminho_do_csv, sep=';', encoding='utf-8')

            # Tratamento dos dados
            df['rentabilidade'] = pd.to_numeric(df['rentabilidade'], errors='coerce')
            df['risco'] = pd.to_numeric(df['risco'], errors='coerce')
            df['quantidade_disponivel'] = pd.to_numeric(df['quantidade_disponivel'], errors='coerce')
            df['preco_unitario'] = pd.to_numeric(df['preco_unitario'], errors='coerce')
            df['investimento_minimo'] = pd.to_numeric(df['investimento_minimo'], errors='coerce')
            df['vencimento'] = pd.to_datetime(df['vencimento'], format='%d/%m/%Y', errors='coerce')
            df.dropna(subset=['rentabilidade', 'risco', 'vencimento'], inplace=True)

            df['quantidade_dias'] = df.apply(
                lambda row: (row['vencimento'] - DATA_ATUAL).days if row['vencimento'] < DATA_RETIRADA else (DATA_RETIRADA - DATA_ATUAL).days,
                axis=1
            )
            df['rentabilidade_total'] = (((1 + (df['rentabilidade'] / 100)) ** (df['quantidade_dias'] / 365)) - 1) * 100
            df['investimento_maximo'] = df['quantidade_disponivel'] * df['preco_unitario']

            # --- Execução do Algoritmo ---
            st.write("\nIniciando a otimização da carteira...")
            progress_bar = st.progress(0)

            melhor_carteira = rodar_algoritmo_genetico(
                ativos_df=df,
                valor_total=VALOR_INVESTIMENTO_TOTAL,
                risco_maximo=RISCO_ACEITAVEL,
                data_retirada=DATA_RETIRADA,
                num_ativos_desejados=QUANTIDADE_ATIVOS,
                qtd_geracoes=GERACOES,
                tamanho_populacao=TAMANHO_POPULACAO,
                taxa_mutacao=TAXA_MUTACAO / 100,
                num_elite= int(TAMANHO_POPULACAO * (NUM_ELITE / 100)),
                progress_bar=progress_bar
            )

            progress_bar.empty() # Remove a barra de progresso ao final

            # --- Exibição dos Resultados Finais ---
            st.write("---")
            exibir_resultados(
                carteira=melhor_carteira,
                ativos_df=df,
                valor_total=VALOR_INVESTIMENTO_TOTAL
            )

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {e}")

if __name__ == "__main__":
    main()
