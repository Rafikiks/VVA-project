"""
analysis.py

Script for the VVA Formula 1 project.

This script performs the following steps:

1. Charge et nettoie les données issues du jeu de données de Formule 1.
2. Calcule des statistiques descriptives et des matrices de corrélation.
3. Entraîne deux modèles de machine learning supervisés (Random Forest et
   régression logistique) pour prédire si un pilote termine sur le podium.
4. Évalue les modèles avec des métriques classiques (accuracy, précision,
   rappel et F1‑score).
5. Applique l'algorithme DBSCAN pour détecter les observations aberrantes
   sur les variables « grille de départ » et « points marqués ».
6. Génère et sauvegarde plusieurs graphiques (barres, courbes, matrices de
   corrélation, importance des variables et clustering).

Les sorties (fichiers PNG et CSV) sont enregistrées dans le répertoire
``plots`` du projet.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import DBSCAN


def load_and_clean(data_dir="csv_cleaned"):
    """Load and preprocess the cleaned F1 dataset.

    Args:
        data_dir (str): Relative path to directory containing cleaned CSV files.

    Returns:
        pd.DataFrame: Merged results with relevant numeric columns converted.
    """
    results = pd.read_csv(os.path.join(data_dir, "results.csv"))
    races = pd.read_csv(os.path.join(data_dir, "races.csv"))
    drivers = pd.read_csv(os.path.join(data_dir, "drivers.csv"))
    constructors = pd.read_csv(os.path.join(data_dir, "constructors.csv"))
    # Convert columns to numeric where appropriate
    results["position"] = pd.to_numeric(results["position"], errors="coerce")
    results["fastestLap"] = pd.to_numeric(results["fastestLap"], errors="coerce")
    results["fastestLapSpeed"] = pd.to_numeric(results["fastestLapSpeed"], errors="coerce")
    # Merge datasets for readability (optional)
    df = results.merge(races[["raceId", "year", "name"]], on="raceId", how="left")
    df = df.merge(drivers[["driverId", "forename", "surname"]], on="driverId", how="left")
    df = df.merge(constructors[["constructorId", "name"]], on="constructorId", how="left", suffixes=("", "_constructor"))
    return df


def plot_correlation_matrix(df, columns, title, filename):
    """Compute and plot a correlation matrix.

    Args:
        df (pd.DataFrame): Dataframe containing numeric columns.
        columns (list): List of column names to include in the matrix.
        title (str): Title of the plot.
        filename (str): Path to save the figure.
    """
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values)
    # Annotate cells
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    ax.set_title(title)
    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def train_models(df, feature_cols, target_col="podium"):
    """Train RandomForest and Logistic Regression models.

    Args:
        df (pd.DataFrame): DataFrame containing features and target.
        feature_cols (list): List of feature column names.
        target_col (str): Name of the target column.

    Returns:
        dict: Metrics for both models.
    """
    # Create target column if not present
    if target_col not in df.columns:
        df[target_col] = (df["position"] <= 3).astype(int)
    # Drop rows with missing values
    data = df.dropna(subset=feature_cols + [target_col])
    X = data[feature_cols]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    metrics = {}
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics["RandomForest"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf),
        "feature_importances": rf.feature_importances_,
    }
    # Logistic Regression with standardization
    log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    metrics["LogisticRegression"] = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr),
    }
    return metrics


def plot_feature_importance(features, importances, filename):
    """Plot feature importances for a model.

    Args:
        features (list): Feature names.
        importances (array): Importance values.
        filename (str): Where to save the plot.
    """
    df_imp = pd.DataFrame({"feature": features, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False)
    fig, ax = plt.subplots()
    ax.bar(df_imp["feature"], df_imp["importance"])
    ax.set_title("Importance des variables (Random Forest)")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Importance relative")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def clustering_analysis(df, feature_cols, output_path):
    """Perform DBSCAN clustering and save a scatter plot showing clusters and outliers.

    Args:
        df (pd.DataFrame): DataFrame containing feature columns.
        feature_cols (list): Two columns used for clustering (e.g., grid and points).
        output_path (str): Path to save the clustering plot.
    """
    assert len(feature_cols) == 2, "Deux colonnes sont requises pour un scatter 2D."
    from sklearn.preprocessing import StandardScaler
    # Drop NaN
    data = df[feature_cols].dropna().copy()
    X = data.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Apply DBSCAN
    clustering = DBSCAN(eps=0.7, min_samples=5)
    labels = clustering.fit_predict(X_scaled)
    data["cluster"] = labels
    # Plot
    fig, ax = plt.subplots()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        marker = "x" if label == -1 else "o"
        ax.scatter(data[feature_cols[0]][mask], data[feature_cols[1]][mask],
                   marker=marker, alpha=0.6, label=("Valeur aberrante" if label == -1 else f"Cluster {label}"))
    ax.set_xlabel(feature_cols[0])
    ax.set_ylabel(feature_cols[1])
    ax.set_title("Clustering DBSCAN: valeurs aberrantes vs données fiables")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return data["cluster"].value_counts().to_dict()


def main():
    """Exécute toutes les étapes d'analyse et génère les fichiers de sortie."""
    # Create output directory
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    # Load data
    df = load_and_clean()
    # Correlation matrix for key variables
    corr_cols = ["grid", "position", "points", "fastestLap", "fastestLapSpeed"]
    plot_correlation_matrix(df, corr_cols,
                            title="Matrice de corrélation des résultats (nettoyés)",
                            filename=os.path.join(output_dir, "heatmap_results_clean.png"))
    # Train models
    feature_cols = ["grid", "fastestLap", "fastestLapSpeed", "points"]
    metrics = train_models(df, feature_cols)
    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_dict({k: {"accuracy": v["accuracy"],
                                             "precision": v["precision"],
                                             "recall": v["recall"],
                                             "f1": v["f1"]} for k, v in metrics.items()},
                                        orient="index")
    metrics_df.to_csv(os.path.join(output_dir, "model_performance_metrics.csv"))
    # Feature importance plot
    rf_importances = metrics["RandomForest"]["feature_importances"]
    plot_feature_importance(feature_cols, rf_importances,
                            filename=os.path.join(output_dir, "rf_feature_importance.png"))
    # Clustering analysis
    clusters_summary = clustering_analysis(df, ["grid", "points"],
                                           output_path=os.path.join(output_dir, "cluster_dbscan_grid_points.png"))
    # Save cluster summary
    cluster_df = pd.DataFrame(list(clusters_summary.items()), columns=["cluster", "count"])
    cluster_df.to_csv(os.path.join(output_dir, "clustering_summary.csv"), index=False)
    print("Analyse terminée. Les fichiers sont enregistrés dans le dossier 'plots'.")


if __name__ == "__main__":
    main()