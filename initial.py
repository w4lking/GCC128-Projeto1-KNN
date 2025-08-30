import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import time

# Carregando o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Para facilitar a visualização, podemos colocar em um DataFrame do Pandas
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y

print("Dataset Iris carregado com sucesso!")
print(df.head())

# Dividir os dados em treino e teste (essencial para avaliação)
# 80% para treino, 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDados de treino: {X_train.shape[0]} amostras")
print(f"Dados de teste: {X_test.shape[0]} amostras")