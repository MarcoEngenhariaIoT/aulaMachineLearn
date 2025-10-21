"""
Classificação da Base Iris com SVM e Visualização
Aula Estácio - Big Data Analytics
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================
# Carrega a base de dados Iris, famosa dataset de classificação de flores
data = load_iris()

# Converte os dados para DataFrame do pandas para melhor manipulação
# As features são: sepal length, sepal width, petal length, petal width (em cm)
iris = pd.DataFrame(data['data'], columns=data['feature_names'])

# Variável target contém as classes (0=setosa, 1=versicolor, 2=virginica)
target = data.target

# MODELAGEM COM SUPPORT VECTOR MACHINE (SVM)
# =============================================
# Instancia o modelo SVM com gamma='auto' para configuração automática do parâmetro
svc = SVC(gamma='auto')

# AVALIAÇÃO DO MODELO COM CROSS-VALIDATION
# =============================================
# Realiza validação cruzada com 10 folds para avaliar a acurácia do modelo
# Cross-validation é importante para evitar overfitting e ter uma avaliação mais robusta
cv_result = cross_val_score(svc, iris, target, cv=10, scoring='accuracy')

# Exibe a acurácia média do modelo em porcentagem
print('Acurácia com cross validation', cv_result.mean()*100)

# TREINAMENTO DO MODELO FINAL
# =============================================
# Treina o modelo SVM com todos os dados disponíveis
svc.fit(iris, target)

# Faz uma predição para um exemplo específico de flor
# [6.9, 2.8, 6.1, 2.3] representa uma flor com características específicas
svc.predict([[6.9,2.8,6.1,2.3]])

# PRIMEIRA VISUALIZAÇÃO: GRÁFICO DE DISPERSÃO
# =============================================
# Cria um scatter plot mostrando a relação entre comprimento e largura da sépala
# As cores (c=target) representam as diferentes espécies de iris
plt.scatter(iris['sepal length (cm)'], iris['sepal width (cm)'], c=target)
plt.title('iris')  # Título do gráfico
plt.show()  # Exibe o gráfico

# PREPARAÇÃO PARA A SEGUNDA VISUALIZAÇÃO (FRONTEIRA DE DECISÃO)
# =============================================
# Calcula os limites dos eixos para criar uma malha de pontos
x0_min, x0_max = iris['sepal length (cm)'].min(), iris['sepal length (cm)'].max()
x1_min, x1_max = iris['petal width (cm)'].min(), iris['petal width (cm)'].max()

# Calcula largura e altura dos intervalos
w = x0_max - x0_min
h = x1_max - x1_min

# Cria uma malha de pontos que cobre toda a área do gráfico com uma margem de 10%
# np.meshgrid cria coordenadas para todos os pontos da malha
x0, x1 = np.meshgrid(np.linspace(x0_min-.1*w, x0_max+.1*w, 300), 
                     np.linspace(x1_min-.1*h, x1_max+.1*h, 300))

# SEGUNDO TREINAMENTO PARA VISUALIZAÇÃO 2D
# =============================================
# Treina um NOVO modelo SVM usando apenas duas features para permitir visualização 2D
# Seleciona apenas comprimento da sépala e largura da pétala
svc.fit(iris[['sepal length (cm)', 'petal width (cm)']], target)

# PREVISÕES PARA A MALHA
# =============================================
# Faz predições para todos os pontos da malha criada
# np.c_ concatena as coordenadas x0 e x1 em um array 2D
# reshape(-1, 1) transforma os arrays em formato adequado para predição
ypred = svc.predict(np.c_[x0.reshape(-1, 1), x1.reshape(-1, 1)])

# Reformata as predições para o formato da malha (300x300)
ypred = ypred.reshape(x0.shape)

# SEGUNDA VISUALIZAÇÃO: MAPA DE DECISÃO
# =============================================
# Cria um gráfico de contorno preenchido mostrando as regiões de decisão do modelo
plt.contourf(x0, x1, ypred)

# Adiciona os pontos reais do dataset sobre o mapa de decisão
# s=64 controla o tamanho dos pontos, edgecolors='k' adiciona bordas pretas
plt.scatter(iris['sepal length (cm)'], iris['petal width (cm)'], c=target, s=64, edgecolors='k')

plt.title('Iris')  # Título do gráfico
plt.show()  # Exibe o gráfico final

"""
RESULTADOS ESPERADOS:
- Acurácia com cross validation: ~98%
- Primeiro gráfico: Dispersão das flores no espaço sepal length vs sepal width
- Segundo gráfico: Mapa de decisão mostrando como o SVM classifica novas amostras
  baseado apenas em sepal length e petal width

OBSERVAÇÕES:
- O alto valor de acurácia indica que o SVM é eficaz para classificar a base Iris
- A visualização da fronteira de decisão ajuda a entender como o modelo toma decisões
- O exemplo de predição [6.9,2.8,6.1,2.3] provavelmente classifica como virginica (classe 2)
"""