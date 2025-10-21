"""
Implementação Simplificada de Múltiplos Classificadores
Aula Estácio - Big Data Analytics

Este script mostra uma implementação concisa de três algoritmos
com foco na aplicação prática básica
"""

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# DADOS EXEMPLO
# =============================================
# Configuração básica para demonstração
X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

# 1. MLPCLASSIFIER
# =============================================
clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(X_train, y_train)

# Probabilidades (grau de certeza das predições)
print("Probabilidades:", clf.predict_proba(X_test[:1]))

# Predições
print("Predição:", clf.predict(X_test[:5]))

# Acurácia
print("Acurácia:", clf.score(X_test, y_test))

# 2. SVC COM PIPELINE
# =============================================
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)

print("Predição:", clf.predict(X_test[:5]))
print("Acurácia:", clf.score(X_test, y_test))

# 3. DECISION TREE COM CROSS VALIDATION
# =============================================
clf = DecisionTreeClassifier(random_state=0)

# Validação cruzada para avaliação robusta
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Cross Validation Scores:", scores)