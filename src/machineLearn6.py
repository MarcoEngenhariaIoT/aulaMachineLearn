"""
Árvore de Decisão com Visualização Detalhada
Aula Estácio - Big Data Analytics

Este script demonstra:
- Criação e análise de DataFrame
- Treinamento de árvore de decisão
- Visualização da árvore
- Análise de importância das features
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# PREPARAÇÃO DOS DADOS
# =============================================
# Cria dados sintéticos para classificação
X, y = make_classification(n_samples=100, random_state=1)

# Divisão estratificada dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

# DATAFRAME PARA ANÁLISE
# =============================================
# Converte para DataFrame e adiciona target
tdf = pd.DataFrame(X_train)
tdf['target'] = y_train
tdf['target'] = tdf['target'].astype('str')

print("DataFrame com dados de treino:")
print(tdf.head())
print(f"\nShape do DataFrame: {tdf.shape}")

# MODELO DE ÁRVORE DE DECISÃO
# =============================================
# Cria árvore com profundidade limitada para melhor visualização
clf = DecisionTreeClassifier(random_state=1, max_depth=3)
clf.fit(X_train, y_train)

# VISUALIZAÇÃO DA ÁRVORE
# =============================================
plt.figure(figsize=(20, 10))
tree.plot_tree(clf, 
               filled=True,  # Preenche nós com cores
               feature_names=[f'Feature_{i}' for i in range(X.shape[1])],  # Nomes das features
               class_names=['Class_0', 'Class_1'],  # Nomes das classes
               rounded=True,  # Bordas arredondadas
               fontsize=10)   # Tamanho da fonte
plt.title("Árvore de Decisão - Classificação")
plt.show()

# AVALIAÇÃO DO MODELO
# =============================================
print(f"\n Acurácia do modelo: {clf.score(X_test, y_test):.2%}")

# ANÁLISE DE IMPORTÂNCIA DAS FEATURES
# =============================================
# Mostra quais features são mais relevantes para a árvore
feature_importance = pd.DataFrame({
    'feature': [f'Feature_{i}' for i in range(X.shape[1])],
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n Importância das features (top 5):")
print(feature_importance.head())