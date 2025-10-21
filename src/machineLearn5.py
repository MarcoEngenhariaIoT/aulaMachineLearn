"""
Análise Exploratória e Classificação com SVC
Aula Estácio - Big Data Analytics

Este script demonstra:
- Criação de DataFrame para análise exploratória
- Estatísticas descritivas dos dados
- Aplicação do SVC com pipeline de pré-processamento
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# CRIAÇÃO DE DADOS ARTIFICIAIS
# =============================================
# Gera dataset sintético para classificação
X, y = make_classification(n_samples=100, random_state=1)

# DIVISÃO TREINO/TESTE
# =============================================
# Separa dados mantendo distribuição balanceada das classes
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

# CRIAÇÃO DO DATAFRAME PARA ANÁLISE
# =============================================
# Converte array numpy em DataFrame pandas para melhor manipulação
tdf = pd.DataFrame(X_train)

# Adiciona a variável target ao DataFrame
tdf['target'] = y_train

# Converte target para string para facilitar análise categórica
tdf['target'] = tdf['target'].astype('str')

# ANÁLISE EXPLORATÓRIA
# =============================================
print("DataFrame com dados de treino:")
print(tdf.head())  # Mostra primeiras linhas
print(f"\nShape do DataFrame: {tdf.shape}")  # Dimensões do dataset

# ESTATÍSTICAS DAS CLASSES
# =============================================
print(f"\nValores únicos na target: {tdf['target'].unique()}")
print(f"\nContagem por classe:")
print(tdf['target'].value_counts())  # Distribuição das classes

# CLASSIFICAÇÃO COM SVC
# =============================================
from sklearn.svm import SVC

# Cria pipeline com normalização e modelo SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

# Treina o modelo
clf.fit(X_train, y_train)

# Predições e avaliação
print("Predição:", clf.predict(X_test[:5,:]))  # Prediz primeiras 5 amostras
print("Acurácia:", clf.score(X_test, y_test))  # Acuracia geral