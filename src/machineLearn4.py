"""
Aplicação Prática do MLPClassifier
Aula Estácio - Big Data Analytics

Este script demonstra o uso prático do MLPClassifier para:
- Treinamento do modelo
- Realização de predições
- Avaliação da acurácia
"""

from sklearn.neural_network import MLPClassifier

# INSTANCIAÇÃO E TREINAMENTO DO MLPClassifier
# =============================================
# Cria e treina o modelo de rede neural
# Assume que x_train e y_train já foram definidos anteriormente
clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(x_train, y_train)

# REALIZAÇÃO DE PREDIÇÕES
# =============================================

# Predição para uma única amostra
# x_test[:1] seleciona a primeira amostra do conjunto de teste
print('Detalhes predição:', clf.predict(x_test[:1]))

# Predição para as primeiras 5 amostras
# x_test[:5, :] seleciona as 5 primeiras amostras com todas as features
print('Predição:', clf.predict(x_test[:5, :]))

# AVALIAÇÃO DO MODELO
# =============================================
# Calcula a acurácia do modelo no conjunto de teste
# A acurácia representa a proporção de predições corretas
print('Modelo Acuradoracia:', clf.score(x_test, y_test))