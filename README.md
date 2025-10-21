# Machine Learning - Big Data Analytics

**Repositório de Códigos da Disciplina Big Data Analytics - Estácio**

Este repositório contém implementações práticas de algoritmos de Machine Learning desenvolvidas como parte das atividades da disciplina de Big Data Analytics da Universidade Estácio de Sá. Os códigos aqui presentes são materiais de aula e exercícios práticos.

## Aviso Importante

**Estes são códigos de aula desenvolvidos durante o curso de Big Data Analytics da Estácio. Foram criados para fins educacionais e de aprendizado, seguindo as orientações e materiais fornecidos pela instituição.**

## Ambiente de Desenvolvimento

- **IDE**: Visual Studio Code
- **Terminal**: PowerShell
- **Python**: 3.12
- **Ambiente**: Virtual Environment (venv)

## Estrutura do Projeto

```
├── src/                    # Códigos fonte principais
│   ├── machineLearn1.py    # Classificação Iris com SVM
│   ├── machineLearn2.py    # Árvore de decisão para Iris
│   ├── machineLearn3.py    # Comparação de classificadores
│   ├── machineLearn4.py    # Aplicação do MLPClassifier
│   ├── machineLearn5.py    # Análise exploratória e SVC
│   ├── machineLearn6.py    # Árvore com visualização detalhada
│   ├── teste.py            # Visualização de fronteira de decisão
│   ├── teste2.py           # Comparação MLP vs SVC
│   ├── teste3.py           # Análise comparativa avançada
│   ├── teste4.py           # Comparação de três algoritmos
│   └── teste5.py           # Implementação simplificada
├── .venv/                  # Ambiente virtual Python
├── requirements.txt        # Dependências do projeto
├── notas.txt              # Anotações e observações
└── .gitignore             # Arquivos ignorados pelo Git
```

## Configuração do Ambiente

### 1. Criar e Ativar Ambiente Virtual

```powershell
# No PowerShell do VSCode
python -m venv .venv

# Ativar o ambiente virtual
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar Dependências

```powershell
pip install -r requirements.txt
```

### 3. Verificar Instalação

```powershell
python --version
pip list
```

## Conteúdo dos Arquivos

### Arquivos Principais

**machineLearn1.py** - Classificação da base Iris utilizando Support Vector Machine (SVC) com validação cruzada e visualização dos resultados.

**machineLearn2.py** - Implementação de árvore de decisão para a base Iris, incluindo validação cruzada e visualização da estrutura da árvore.

**machineLearn3.py** - Comparação entre três algoritmos de classificação: MLPClassifier, SVC com pipeline e DecisionTreeClassifier.

**machineLearn4.py** - Aplicação prática do MLPClassifier para classificação, demonstrando treinamento, predição e avaliação.

**machineLearn5.py** - Análise exploratória de dados com pandas e classificação utilizando SVC com pipeline de pré-processamento.

**machineLearn6.py** - Árvore de decisão com visualização detalhada, incluindo análise de importância das features.

### Arquivos de Teste

**teste.py** - Visualização da fronteira de decisão do MLPClassifier em problemas 2D.

**teste2.py** - Comparação direta entre MLPClassifier e SVC.

**teste3.py** - Análise comparativa avançada com métricas detalhadas e validação cruzada.

**teste4.py** - Comparação abrangente entre três algoritmos com diferentes conjuntos de dados.

**teste5.py** - Implementação simplificada para demonstração rápida.

## Como Executar

### Execução Individual

```powershell
# Navegar para a pasta src
cd src

# Executar arquivo específico
python machineLearn1.py
python teste4.py
```

## Algoritmos Implementados

### Técnicas de Classificação

- Support Vector Machine (SVC)
- Multi-layer Perceptron (MLPClassifier)
- Árvore de Decisão (DecisionTreeClassifier)

### Métodos de Validação

- Validação Cruzada (Cross Validation)
- Divisão Treino/Teste
- Pipeline de pré-processamento
- Métricas de avaliação (acurácia, precisão, recall)

### Visualização

- Fronteiras de decisão
- Estruturas de árvores
- Gráficos de dispersão
- Análise de importância de features

## Bases de Dados Utilizadas

- **Iris Dataset**: Problema clássico de classificação de espécies de flores
- **Dados Sintéticos**: Gerados através da função `make_classification` do scikit-learn

## Resultados Esperados

### Acurácias Típicas

- Base Iris: ~98% com SVM
- MLPClassifier: Bom desempenho em problemas não-lineares
- SVC: Excelente com dados normalizados
- Árvore de Decisão: Interpretável mas sujeito a overfitting

### Visualizações Geradas

- Gráficos de fronteira de decisão
- Estruturas de árvores de decisão
- Dispersão de dados com cores de classe
- Mapas de importância de features

## requirements.txt

```
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
numpy>=1.24.0
```

### Problemas com o Ambiente Virtual

```powershell
# Recriar ambiente virtual
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Erro de Importação de Módulos

- Certifique-se de estar na pasta raiz do projeto
- Verifique se o ambiente virtual está ativado
- Confirme a instalação das dependências

## Contexto Acadêmico

Este repositório foi desenvolvido como material de apoio para a disciplina **Big Data Analytics** da **Universidade Estácio de Sá**, abordando:

- Fundamentos de Machine Learning
- Implementação prática de algoritmos
- Análise e interpretação de resultados
- Boas práticas em ciência de dados

_Configurado para Python 3.12, VSCode e PowerShell_
