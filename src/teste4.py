"""
Comparação de Três Algoritmos de Classificação
Aula Estácio - Big Data Analytics

Este script compara o desempenho de três algoritmos diferentes:
- MLPClassifier (Redes Neurais)
- SVC (Máquinas de Vetor de Suporte)
- DecisionTreeClassifier (Árvore de Decisão)
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ========== PREPARAÇÃO DOS DADOS ==========
# Dataset com características mistas (informativas e redundantes)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, 
                          n_redundant=10, random_state=42)

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                   random_state=42, stratify=y)

print(f"Dados: {X.shape}, Treino: {X_train.shape}, Teste: {X_test.shape}")

# ========== 1. MLPCLASSIFIER ==========
print("\n" + "="*50)
print("1. MLPClassifier")
print("="*50)

clf_mlp = MLPClassifier(random_state=1, max_iter=300)
clf_mlp.fit(X_train, y_train)

# Probabilidades de classe (útil para análise de confiança)
print("Probabilidades (2 primeiras amostras):")
print(clf_mlp.predict_proba(X_test[:2]))

# Predições
print("Previsões (5 primeiras amostras):")
print(clf_mlp.predict(X_test[:5]))

print(f"Acurácia do Modelo: {clf_mlp.score(X_test, y_test):.2%}")

# ========== 2. SVC COM PIPELINE ==========
print("\n" + "="*50)
print("2. SVC com Pipeline")
print("="*50)

clf_svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf_svc.fit(X_train, y_train)

print("Previsões (5 primeiras amostras):")
print(clf_svc.predict(X_test[:5]))

print(f"Acurácia do Modelo: {clf_svc.score(X_test, y_test):.2%}")

# ========== 3. DECISIONTREECLASSIFIER ==========
print("\n" + "="*50)
print("3. DecisionTreeClassifier com Cross Validation")
print("="*50)

clf_dt = DecisionTreeClassifier(random_state=0)

# Validação cruzada para avaliação mais confiável
cv_scores = cross_val_score(clf_dt, X_train, y_train, cv=10)

print("Scores de Cross Validation (10 folds):")
print([f"{score:.4f}" for score in cv_scores])
print(f"Média CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Treino final e teste
clf_dt.fit(X_train, y_train)
test_score = clf_dt.score(X_test, y_test)
print(f"Acurácia no Teste: {test_score:.2%}")

# ========== COMPARAÇÃO FINAL ==========
print("\n" + "="*50)
print("COMPARAÇÃO FINAL DOS MODELOS")
print("="*50)

models = {
    'MLPClassifier': clf_mlp.score(X_test, y_test),
    'SVC': clf_svc.score(X_test, y_test),
    'DecisionTree': clf_dt.score(X_test, y_test)
}

for model_name, score in models.items():
    print(f"{model_name:20} → {score:.2%}")