"""
Implementação de Diferentes Algoritmos de Classificação
Aula Estácio - Big Data Analytics

Este demonstra a instanciação de três diferentes algoritmos de ML:
- Redes Neurais (MLPClassifier)
- Máquinas de Vetor de Suporte (SVC) com Pipeline
- Árvore de Decisão
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# PREPARAÇÃO DOS DADOS ARTIFICIAIS
# =============================================
# Gera dados sintéticos para classificação binária
# n_samples=100: cria 100 amostras
# random_state=1: garante reprodutibilidade dos resultados
x, y = make_classification(n_samples=100, random_state=1)

# DIVISÃO DOS DADOS
# =============================================
# Separa os dados em conjuntos de treino e teste
# stratify=y: mantém a proporção das classes em ambos os conjuntos
# random_state=1: garante divisão consistente
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)

# INSTANCIAÇÃO DOS MODELOS
# =============================================

# 1. REDES NEURAIS (MLPClassifier)
# =============================================
# Multi-layer Perceptron: rede neural artificial para classificação
# random_state=1: reprodutibilidade
# max_iter=300: número máximo de épocas de treinamento
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(random_state=1, max_iter=300)

# 2. MÁQUINAS DE VETOR DE SUPORTE (SVC)
# =============================================
# Pipeline que inclui pré-processamento (StandardScaler) e modelo SVC
# StandardScaler: normaliza os dados (média=0, desvio padrão=1)
# SVC: Support Vector Classification com gamma='auto'
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

# 3. ÁRVORE DE DECISÃO
# =============================================
# Modelo de árvore de decisão para classificação
# random_state=0: semente para aleatoriedade
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

# VISUALIZAÇÃO DA ÁRVORE DE DECISÃO
# =============================================
# Importa bibliotecas para plotagem
import matplotlib.pyplot as plt
from sklearn import tree

# Treina a árvore com dados de treino e plota a estrutura
# A visualização ajuda a entender como a árvore toma decisões
tree.plot_tree(clf.fit(x_train, y_train))
plt.show()