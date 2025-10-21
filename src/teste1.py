"""
Visualização da Fronteira de Decisão do MLPClassifier
Aula Estácio - Big Data Analytics

Este script demonstra como visualizar a fronteira de decisão
de um modelo MLPClassifier em um problema de classificação 2D
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

# CRIAÇÃO DE DADOS 2D PARA VISUALIZAÇÃO
# =============================================
# Gera dados com apenas 2 features para permitir visualização 2D
# n_features=2: apenas duas dimensões para plotagem
# n_informative=2: ambas as features são informativas
# n_redundant=0: sem features redundantes
x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, random_state=1)

# Divisão dos dados
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)

# TREINAMENTO DO MODELO
# =============================================
clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(x_train, y_train)

# PREPARAÇÃO DA MALHA PARA PLOTAGEM
# =============================================
# Define os limites do gráfico com margens
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

# Cria grid de pontos para avaliar o modelo em toda a área
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# PREDIÇÃO SOBRE A MALHA
# =============================================
# Prediz a classe para cada ponto do grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)  # Reformata para dimensões do grid

# PLOTAGEM DA FRONTEIRA DE DECISÃO
# =============================================
# contourf: plota áreas coloridas para cada classe
# scatter: plota os pontos originais
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.8)
plt.title("Fronteira de Decisão do MLPClassifier")
plt.show()