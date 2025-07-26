import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

# --- Functions ---

@st.cache
def load_data(path="wdbc.csv"):
    df = pd.read_csv(path)
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
    return df

def preprocess_data(df):
    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine (SVM)": SVC(probability=True),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

def style_classification_report(df):
    def highlight_cells(val):
        color = ""
        if isinstance(val, (float, int)):
            if val >= 0.95:
                color = "background-color: #c6f5c6"
            elif val >= 0.9:
                color = "background-color: #fff3b0"
            else:
                color = "background-color: #f5c6c6"
        return color

    return df.style.map(highlight_cells, subset=["precision", "recall", "f1-score"])

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    cm = confusion_matrix(y_test, y_pred)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    return acc, report_df, cm, y_score

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

def plot_roc_curve(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

def plot_precision_recall_curve(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color="b", label=f"PR AUC = {pr_auc:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    st.pyplot(fig)
    
def plot_boxplots(df, features):
    if len(features) == 0:
        st.info("Please select at least one feature to display boxplots.")
        return

    fig, axs = plt.subplots(1, len(features), figsize=(6 * len(features), 4))
    if len(features) == 1:
        axs = [axs]
    for i, feature in enumerate(features):
        sns.boxplot(x=df["diagnosis"].map({0: "Benign", 1: "Malignant"}), y=df[feature], ax=axs[i])
        axs[i].set_title(f"Distribution of {feature} by Class")
        axs[i].set_xlabel("Class")
        axs[i].set_ylabel(feature)
    plt.tight_layout()
    st.pyplot(fig)
    
# Random forest and gradient boosting only  
def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance is not available for this model.")
        return

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10], ax=ax, palette="viridis")
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)
    
def plot_top_feature_histograms(df, feature_names, target_col="diagnosis", top_n=3):
    fig, axs = plt.subplots(1, top_n, figsize=(5 * top_n, 4))
    top_features = feature_names[:top_n]
    for i, feature in enumerate(top_features):
        sns.histplot(data=df, x=feature, hue=target_col, bins=30, kde=True, ax=axs[i], palette=["green", "red"])
        axs[i].set_title(f"Distribution of {feature}")
    plt.tight_layout()
    st.pyplot(fig)
    
def plot_pairplot(df, features, target_col="diagnosis"):
    if len(features) < 2:
        st.info("Select at least 2 features for pairplot.")
        return
    sns_plot = sns.pairplot(df[features + [target_col]], hue=target_col, palette=["green", "red"])
    st.pyplot(sns_plot)
    
def plot_correlation_matrix(df, features):
    corr = df[features].corr()

    # Ajusta el tamaño según la cantidad de variables
    n = len(features)
    fig_size = max(1.2 * n, 10)  # Asegura un tamaño mínimo decente

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,  # celdas cuadradas
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.6, "label": "Correlation"},
        annot_kws={"size": 10, "weight": "bold", "color": "black"}
    )

    ax.set_title("Correlation Matrix Heatmap", fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)
    
def resumen_correlaciones(df, features, umbral_alto=0.8, umbral_bajo=0.3):
    corr_matrix = df[features].corr().abs()  # Módulo para centrarse en la fuerza
    upper_tri = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
    
    correlaciones_altas = (
        upper_tri.stack()
        .reset_index()
        .rename(columns={"level_0": "Variable 1", "level_1": "Variable 2", 0: "Correlación"})
        .query("Correlación >= @umbral_alto")
        .sort_values(by="Correlación", ascending=False)
    )

    correlaciones_bajas = (
        upper_tri.stack()
        .reset_index()
        .rename(columns={"level_0": "Variable 1", "level_1": "Variable 2", 0: "Correlación"})
        .query("Correlación <= @umbral_bajo")
        .sort_values(by="Correlación", ascending=True)
    )

    return correlaciones_altas, correlaciones_bajas   
    
# --- App ---

def main():
    st.title("🧬 Breast Cancer ML Classifier with Advanced Visualization")

    df = load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = preprocess_data(df)
    models = get_models()

    selected_model_name = st.selectbox("Select a Machine Learning model:", list(models.keys()))
    model = models[selected_model_name]

    model = train_model(model, X_train_scaled, y_train)

    acc, report_df, cm, y_score = evaluate_model(model, X_test_scaled, y_test)

    st.subheader(f"📊 Results for {selected_model_name}")
    st.write(f"✅ **Accuracy:** {acc:.4f}")

    st.subheader("🔍 Classification Report (Table)")
    st.dataframe(style_classification_report(report_df))

    st.subheader("📌 Confusion Matrix")
    plot_confusion_matrix(cm)

    st.subheader("📈 ROC Curve")
    if y_score is not None:
        plot_roc_curve(y_test, y_score)
    else:
        st.write("The selected model does not support probability scores for ROC curve calculation.")

    st.subheader("🎯 Precision-Recall Curve")
    if y_score is not None:
        plot_precision_recall_curve(y_test, y_score)
    else:
        st.write("The selected model does not support scoring for the Precision-Recall curve.")
        
    st.subheader("📊 Boxplots")
    selected_box_features = st.multiselect("Select up to 2 features for boxplots:", options=feature_names.tolist(), default=feature_names.tolist()[:2])
    if len(selected_box_features) > 2:
        st.warning("Select up to 2 features only.")
        selected_box_features = selected_box_features[:2]
    plot_boxplots(df, selected_box_features)
    
    st.subheader("📈 Feature Importance")
    if selected_model_name in ["Random Forest", "Gradient Boosting"]:
        plot_feature_importance(model, feature_names)
    else:
        st.info("Feature importance not available for this model.")
    
    st.subheader("📊 Histograms of Top Features")
    plot_top_feature_histograms(df, feature_names)
    
    st.subheader("🔗 Pairplot")
    selected_pairplot_features = st.multiselect("Select features for pairplot (min 2):", options=feature_names.tolist(), default=feature_names.tolist()[:3])
    if len(selected_pairplot_features) >= 2:
        plot_pairplot(df, selected_pairplot_features)
    else:
        st.info("Select at least 2 features for pairplot.")
    
    st.subheader("📉 Correlation Matrix")
    plot_correlation_matrix(df, feature_names)
    altas, bajas = resumen_correlaciones(df, feature_names)

    st.subheader("Correlaciones Altas (>= 0.8)")
    st.dataframe(altas)
    
    st.subheader("Correlaciones Bajas (<= 0.3)")
    st.dataframe(bajas)

if __name__ == "__main__":
    main()
