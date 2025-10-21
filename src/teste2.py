"""
Comparação entre MLPClassifier e SVC
Aula Estácio - Big Data Analytics

Este script compara o desempenho de dois algoritmos:
- MLPClassifier (Redes Neurais)
- SVC (Máquinas de Vetor de Suporte) com pré-processamento
"""

# Importações
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# PREPARAÇÃO DOS DADOS
# =============================================
# Gera dados com 4 features para comparação mais realista
X, y = make_classification(n_samples=100, n_features=4, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, random_state=42)

# Divisão padrão 70/30 para treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MODELO 1: MLPClassifier
# =============================================
mlp_clf = MLPClassifier(random_state=1, max_iter=300)
mlp_clf.fit(X_train, y_train)
mlp_pred = mlp_clf.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)

# MODELO 2: SVC COM PIPELINE
# =============================================
# Pipeline inclui normalização (importante para SVC)
svc_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc_clf.fit(X_train, y_train)
svc_pred = svc_clf.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_pred)

# COMPARAÇÃO DOS RESULTADOS
# =============================================
print(f"Acurácia do MLPClassifier: {mlp_accuracy:.2f}")
print(f"Acurácia do SVC: {svc_accuracy:.2f}")