
# FASE 1: DATA EXPLORATION & PREPROCESSING

# 1. Importazione librerie

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# 2. Caricamento dataset Iris

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)


# 3. Analisi esplorativa

print("\n===== DATASET IRIS =====")

print("\nPrime 5 osservazioni:")
print(df.head())

print("\nDimensioni dataset:")
print(df.shape)

print("\nInformazioni dataset:")
df.info()

print("\nStatistiche descrittive:")
print(df.describe())

print("\nValori mancanti:")
print(df.isnull().sum())

print("\nDuplicati:")
print(df.duplicated().sum())


# 4. Selezione delle feature

X = df.copy()


# 5. Standardizzazione

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)


# 6. Verifica standardizzazione

print("\n===== VERIFICA STANDARDIZZAZIONE =====")

print("\nMedie:")
print(X_scaled.mean())

print("\nDeviazioni standard:")
print(X_scaled.std(ddof=0))

print("\nPrime 5 osservazioni standardizzate:")
print(X_scaled.head())

print("\n===== FINE FASE 1 =====")

# Visualizzazione delle righe duplicate
print("\nRighe duplicate:")
print(df[df.duplicated(keep=False)])


# FASE 2: STIMA DI K CON DBSCAN

from sklearn.cluster import DBSCAN

# Parametri di analisi
eps_values = [0.35, 0.45, 0.55]
min_samples = 5

# Lista per memorizzare i risultati
dbscan_results = []

# Memorizzazione delle etichette per ogni eps
dbscan_labels_by_eps = {}

print("\n===== ANALISI DBSCAN =====")

# Esecuzione iterativa
for eps in eps_values:

    # Inizializzazione DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    # Applicazione algoritmo
    labels = dbscan.fit_predict(X_scaled)
    print("\nDistribuzione osservazioni:")
    print(pd.Series(labels).value_counts().sort_index())
    

    # Individuazione delle etichette
    unique_labels = set(labels)
    
    # Salvataggio delle etichette dell'esecuzione
    dbscan_labels_by_eps[eps] = labels.copy()

    # Calcolo numero cluster (escludendo il rumore -1)
    n_clusters = len(unique_labels - {-1})

    # Calcolo numero punti rumorosi
    n_noise = np.sum(labels == -1)

    # Memorizzazione risultati
    dbscan_results.append({
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise
    })

    # Stampa risultati
    print(f"\nEsecuzione con eps = {eps}")
    print("Numero cluster:", n_clusters)
    print("Punti rumorosi:", n_noise)


# Creazione tabella riepilogativa
results_df = pd.DataFrame(dbscan_results)

print("\n===== TABELLA RIEPILOGATIVA =====")
print(results_df.to_string(index=False))

print("\n===== FINE FASE 2 =====")


# VISUALIZZAZIONE DBSCAN

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

for ax, eps in zip(axes, eps_values):

    labels = dbscan_labels_by_eps[eps]

    # Individuazione dei cluster
    unique_labels = sorted(set(labels) - {-1})

    # Disegno dei cluster
    for cluster_id in unique_labels:

        mask = labels == cluster_id

        ax.scatter(
            X_scaled.iloc[mask, 0],
            X_scaled.iloc[mask, 1],
            color=plt.cm.tab10(cluster_id % 10),
            s=45,
            alpha=0.75,
            label=f"Cluster {cluster_id}"
        )

    # Disegno dei punti rumorosi
    noise_mask = labels == -1

    ax.scatter(
        X_scaled.iloc[noise_mask, 0],
        X_scaled.iloc[noise_mask, 1],
        c="black",
        marker="x",
        s=45,
        label="Rumore (-1)"
    )

    # Titolo e assi
    ax.set_title(f"DBSCAN - eps = {eps}")
    ax.set_xlabel("Sepal Length (standardizzata)")
    ax.set_ylabel("Sepal Width (standardizzata)")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("DBSCAN Clustering - Dataset Iris", fontsize=15)

plt.tight_layout()
plt.show()


# FASE 3: STIMA DI K CON ELBOW METHOD

from sklearn.cluster import KMeans

# Definizione intervallo di K
K_values = range(1, 11)

# Lista per memorizzare le inerzie
inertias = []

print("\n===== ANALISI ELBOW METHOD =====")

# Esecuzione K-Means per ogni valore di K
for k in K_values:

    # Inizializzazione K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Addestramento del modello
    kmeans.fit(X_scaled)

    # Memorizzazione Inertia
    inertias.append(kmeans.inertia_)

    print(f"K = {k} | Inertia = {kmeans.inertia_:.4f}")


# Creazione tabella riepilogativa
elbow_results = pd.DataFrame({
    "K": list(K_values),
    "Inertia": inertias
})

print("\n===== TABELLA INERTIA =====")
print(elbow_results.to_string(index=False))


# VISUALIZZAZIONE ELBOW METHOD

# K candidato individuato preliminarmente con DBSCAN
K_scelto = 3

plt.figure(figsize=(9, 6))

# Grafico Elbow
plt.plot(
    list(K_values),
    inertias,
    marker="o",
    linestyle="-",
    label="Inertia (WCSS)"
)

# Linea verticale sul K candidato
plt.axvline(
    x=K_scelto,
    color="red",
    linestyle="--",
    label=f"K candidato = {K_scelto}"
)

plt.xticks(list(K_values))
plt.xlabel("Numero di cluster (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method - Dataset Iris")

plt.legend()
plt.grid(True)

plt.show()

print("\nK candidato da DBSCAN:", K_scelto)
print("\n===== FINE FASE 3 =====")

# FASE 4: APPLICAZIONE FINALE K-MEANS #

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Definizione del numero finale di cluster
K_final = 3

# 2. Inizializzazione modello K-Means
kmeans_final = KMeans(
    n_clusters=K_final,
    random_state=42,
    n_init=10
)

# 3. Addestramento e assegnazione cluster
clusters_final = kmeans_final.fit_predict(X_scaled)

# 4. Distribuzione delle osservazioni
print("\n===== K-MEANS DEFINITIVO =====")

print("\nNumero di cluster:", K_final)

print("\nDistribuzione osservazioni:")
print(pd.Series(clusters_final).value_counts().sort_index())

# 5. Estrazione dei centroidi
centroids = kmeans_final.cluster_centers_

print("\nCentroidi dei cluster:")
print(centroids)

# 6. Calcolo Silhouette Score
silhouette_avg = silhouette_score(
    X_scaled,
    clusters_final
)

print("\nSilhouette Score:", round(silhouette_avg, 4))


# VISUALIZZAZIONE DEI CLUSTER

plt.figure(figsize=(10, 7))

# Rappresentazione delle osservazioni
plt.scatter(
    X_scaled.iloc[:, 0],
    X_scaled.iloc[:, 1],
    c=clusters_final,
    cmap="viridis",
    s=50,
    alpha=0.7,
    label="Osservazioni"
)

# Rappresentazione dei centroidi
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    c="red",
    edgecolors="black",
    linewidths=1.5,
    label="Centroidi"
)

plt.title("K-Means Clustering - Dataset Iris (K=3)")
plt.xlabel("Sepal Length (standardizzata)")
plt.ylabel("Sepal Width (standardizzata)")

plt.legend()
plt.grid(True)

plt.show()

print("\n===== FINE FASE 4 =====")