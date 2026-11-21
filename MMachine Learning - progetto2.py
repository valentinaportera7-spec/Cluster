import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_linnerud
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

data = load_linnerud(as_frame=True)

df = data.frame

print(df.head())

print(df.describe())

X = data.data
y = data.target

print(X.head())
print(y.head())

#
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

print("Feature standardizzate:")
print(X_scaled)

print("Target standardizzati:")
print(y_scaled)

# FASE 2 - PCA SUI TARGET #

pca_y = PCA(n_components=1)

y_pca = pca_y.fit_transform(y_scaled)

print("Target ottenuto con PCA:")
print(y_pca)

print("Varianza spiegata da PC1:")
print(pca_y.explained_variance_ratio_)

y_waist = y_scaled[:, 1]

print("Target selezionato manualmente (Waist):")
print(y_waist)

# FASE 3 - Definizione dei modelli #

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}
# Scenari di target

targets = {
    "PCA": y_pca.ravel(),
    "Waist": y_waist
}
results = {}

for target_name, target in targets.items():

    for model_name, model in models.items():

        model.fit(X_scaled, target)

        y_pred = model.predict(X_scaled)

        mse = mean_squared_error(target, y_pred)
        r2 = r2_score(target, y_pred)

        results[(target_name, model_name)] = {
            "MSE": mse,
            "R2": r2
        }

        print(f"\nScenario: {target_name}")
        print(f"Modello: {model_name}")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")

# FASE 4 - Analisi e Visualizzazione #

pca_X = PCA(n_components=2)

X_pca = pca_X.fit_transform(X_scaled)

print("Feature trasformate con PCA:")
print(X_pca)

print("Varianza spiegata:")
print(pca_X.explained_variance_ratio_)


# Selezione della sola prima componente delle feature

X_pc1 = X_pca[:, 0].reshape(-1, 1)

# Target PCA

target = y_pca.ravel()


# Creazione del grafico

plt.figure(figsize=(10, 6))

# Scatter plot dei dati

plt.scatter(X_pc1.ravel(), target, color='blue', label='Dati reali')

# Valori X per disegnare le rette

X_line = np.linspace(X_pc1.min(), X_pc1.max(), 100).reshape(-1, 1)

# Definizione dei modelli

models_plot = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

# Addestramento e disegno delle rette

for model_name, model in models_plot.items():

    model.fit(X_pc1, target)

    y_pred = model.predict(X_pc1)

    r2 = r2_score(target, y_pred)

    y_line = model.predict(X_line)

    plt.plot(X_line, y_line,
             label=f'{model_name} (R2={r2:.4f})')

plt.xlabel('PC1 delle Feature')
plt.ylabel('PCA Target (PC1)')
plt.title('Confronto modelli di regressione')
plt.legend()
plt.grid(True)

plt.show()