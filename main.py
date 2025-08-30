import pandas as pd
from collections import Counter
import numpy as np

class KNN:
    """
    Uma implementação do classificador K-Nearest Neighbors (KNN)
    orientada a objetos.
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

# --- Bloco Principal de Execução ---

if __name__ == '__main__':
    # 1. Carregar e preparar os dados
    df = pd.read_csv('data\Iris.csv')
    df = df.drop('Id', axis=1)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # --- SEM EMBARALHAMENTO ---
    # Os dados são usados na ordem original do arquivo.

    # Dividir em treino e teste (70% treino, 30% teste)
    train_size = int(0.7 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 2. Testar o classificador para diferentes valores de k
    k_values = [1, 3, 5, 7]
    
    print("Resultados do KNN Orientado a Objetos (sem embaralhamento):")
    
    for k in k_values:
        model = KNN(k=k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = np.sum(predictions == y_test) / len(y_test)
        print(f'Acurácia para k = {k}: {accuracy:.2f}')