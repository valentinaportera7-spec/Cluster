import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Caricamento dataset

data = load_wine()

X = data.data
y = data.target

# Creazione DataFrame

df = pd.DataFrame(X, columns=data.feature_names)

print(df)

# Statistiche descrittive

print("Statistiche descrittive:")
print(df.describe())

# Tipologia delle colonne

print("\nTipologia delle colonne:")
print(df.dtypes)

# Standardizzazione delle feature

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled.shape)
print(X_scaled[:5])

# Applicazione PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Dimensioni dopo PCA:")
print(X_pca.shape)

print("\nVarianza spiegata:")
print(pca.explained_variance_ratio_)

print("\nVarianza spiegata totale:")
print(sum(pca.explained_variance_ratio_))

# Inizializzazione dei modelli

knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear')

# Addestramento dei modelli

knn.fit(X_pca, y)
svm.fit(X_pca, y)

# Predizioni dei modelli

y_pred_knn = knn.predict(X_pca)
y_pred_svm = svm.predict(X_pca)

# Calcolo Accuracy Score

accuracy_knn = accuracy_score(y, y_pred_knn)
accuracy_svm = accuracy_score(y, y_pred_svm)

# Stampa dei risultati

print("Accuracy KNN:", accuracy_knn)
print("Accuracy SVM:", accuracy_svm)

# Creazione della griglia

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predizioni sulla griglia

Z_knn = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])

# Riportiamo le predizioni alla forma della griglia

Z_knn = Z_knn.reshape(xx.shape)
Z_svm = Z_svm.reshape(xx.shape)

# Creazione dei due subplot

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Grafico KNN

axes[0].contourf(xx, yy, Z_knn, alpha=0.3, cmap=plt.cm.coolwarm)
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
axes[0].set_title(f'KNN (Accuracy: {accuracy_knn:.3f})')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# Grafico SVM

axes[1].contourf(xx, yy, Z_svm, alpha=0.3, cmap=plt.cm.coolwarm)
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
axes[1].set_title(f'SVM (Accuracy: {accuracy_svm:.3f})')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.show()