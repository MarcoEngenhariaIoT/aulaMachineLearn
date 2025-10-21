"""
Implementação de Árvore de Decisão para a Base Iris
Aula Estácio - Big Data Analytics - Tema 6

Este script demonstra:
- Uso do DecisionTreeClassifier
- Validação cruzada para avaliação
- Visualização da estrutura da árvore
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

"""
PROCESSO DE EXPERIMENTAÇÃO COM VALIDAÇÃO CRUZADA
=============================================
A validação cruzada divide os dados em k partes (folds) e 
treina o modelo k vezes, cada vez com um fold diferente como teste
"""

# CONFIGURAÇÃO DO MODELO
# =============================================
# max_depth=3: limita a profundidade para evitar overfitting
# random_state=0: garante reprodutibilidade
clf = DecisionTreeClassifier(max_depth=3, random_state=0)

# CARREGAMENTO DOS DADOS IRIS
# =============================================
iris = load_iris()

# VALIDAÇÃO CRUZADA
# =============================================
# cv=10: divisão em 10 folds para avaliação robusta
# Retorna array com as acurácias de cada fold
cross_val_score(clf, iris.data, iris.target, cv=10)

"""
TREINAMENTO E VISUALIZAÇÃO DA ÁRVORE
=============================================
Após a validação, treinamos o modelo com todos os dados
e visualizamos a estrutura da árvore para interpretabilidade
"""

# TREINAMENTO FINAL
# =============================================
clf.fit(iris.data, iris.target)

# VISUALIZAÇÃO DA ÁRVORE
# =============================================
# filled=True: preenche os nós com cores para melhor visualização
plot_tree(clf, filled=True)
plt.show()