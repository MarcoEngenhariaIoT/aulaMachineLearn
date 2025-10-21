"""
Análise Comparativa Abrangente: MLP vs SVC
Aula Estácio - Big Data Analytics

Este script realiza uma comparação detalhada entre MLPClassifier e SVC,
incluindo métricas avançadas e visualizações
"""

# ========== IMPORTAÇÕES NECESSÁRIAS ==========
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ========== PREPARAÇÃO DOS DADOS ==========
# Dataset mais complexo para teste robusto
X, y = make_classification(
    n_samples=1000,           # Amostra maior para melhor generalização
    n_features=20,            # Mais features para problema mais realista
    n_redundant=2,           # Algumas features redundantes
    n_informative=10,        # Features realmente informativas
    n_clusters_per_class=1,  # Estrutura dos clusters
    random_state=42          # Reprodutibilidade
)

# Divisão estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,           # 30% para teste
    stratify=y,              # Balanceamento mantido
    random_state=42
)

print(f"Shape dos dados: {X.shape}")
print(f"Shape treino: {X_train.shape}, Shape teste: {X_test.shape}")

# ========== CONFIGURAÇÃO DOS MODELOS ==========

# 1. MLPClassifier (REDES NEURAIS)
# =============================================
mlp_model = MLPClassifier(
    random_state=1, 
    max_iter=300,
    hidden_layer_sizes=(100, 50),  # Arquitetura: 2 camadas ocultas
    activation='relu',             # Função de ativação ReLU
    solver='adam'                  # Otimizador Adam
)

# 2. SVC COM PIPELINE
# =============================================
svc_model = make_pipeline(
    StandardScaler(),  # Normalização crítica para SVC
    SVC(gamma='auto', random_state=42)
)

# ========== TREINAMENTO DOS MODELOS ==========
print("\n Treinando MLPClassifier...")
mlp_model.fit(X_train, y_train)

print(" Treinando SVC...")
svc_model.fit(X_train, y_train)

# ========== AVALIAÇÃO COMPARATIVA ==========
mlp_pred = mlp_model.predict(X_test)
svc_pred = svc_model.predict(X_test)

mlp_accuracy = accuracy_score(y_test, mlp_pred)
svc_accuracy = accuracy_score(y_test, svc_pred)

print(f"\n RESULTADOS:")
print(f"MLPClassifier Acurácia: {mlp_accuracy:.2%}")
print(f"SVC Acurácia: {svc_accuracy:.2%}")

# ========== RELATÓRIOS DETALHADOS ==========
print("\n RELATÓRIO MLPClassifier:")
print(classification_report(y_test, mlp_pred))

print("\n RELATÓRIO SVC:")
print(classification_report(y_test, svc_pred))

# ========== VISUALIZAÇÃO 2D ==========
# Cria dados simplificados para visualização
X_vis, y_vis = make_classification(
    n_samples=300,
    n_features=2,           # Apenas 2D para plotagem
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y_vis, test_size=0.3, random_state=42
)

# Modelos para visualização
mlp_vis = MLPClassifier(random_state=1, max_iter=300)
svc_vis = make_pipeline(StandardScaler(), SVC(gamma='auto'))

mlp_vis.fit(X_train_vis, y_train_vis)
svc_vis.fit(X_train_vis, y_train_vis)

# FUNÇÃO PARA PLOTAR FRONTEIRAS DE DECISÃO
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# PLOTAGEM COMPARATIVA
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(mlp_vis, X_test_vis, y_test_vis, 'MLPClassifier - Fronteira de Decisão')

plt.subplot(1, 2, 2)
plot_decision_boundary(svc_vis, X_test_vis, y_test_vis, 'SVC - Fronteira de Decisão')

plt.tight_layout()
plt.show()

# ========== VALIDAÇÃO CRUZADA ==========
from sklearn.model_selection import cross_val_score

# Avaliação mais robusta com cross-validation
mlp_cv_scores = cross_val_score(mlp_model, X, y, cv=5)
svc_cv_scores = cross_val_score(svc_model, X, y, cv=5)

print(f"\n CROSS VALIDATION (5 folds):")
print(f"MLPClassifier: {mlp_cv_scores.mean():.2%} (+/- {mlp_cv_scores.std() * 2:.2%})")
print(f"SVC: {svc_cv_scores.mean():.2%} (+/- {svc_cv_scores.std() * 2:.2%})")