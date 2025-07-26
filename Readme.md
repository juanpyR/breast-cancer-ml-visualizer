# 🧬 Breast Cancer ML Classifier with Advanced Visualization

This project is an interactive web app built with [Streamlit](https://streamlit.io/) that allows users to explore and compare different classification models to predict whether a breast tumor is benign or malignant using the **Wisconsin Diagnostic Breast Cancer dataset** (`wdbc.csv`).

## 🚀 Key Features

- Classification using multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting
- Data preprocessing with feature scaling (`StandardScaler`)
- Model evaluation with:
  - Interactive classification report
  - Confusion matrix
  - ROC curve and AUC score
  - Precision-Recall curve
- Exploratory data analysis:
  - Boxplots
  - Histograms of top features
  - Pairplot
  - Correlation heatmap
  - High and low correlation summaries

## 📦 Requirements

- Python 3.7 or higher
- Install dependencies:

```
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

## 🗂 Project Structure

```
.
├── breast_cancer.py        # Main Streamlit app
├── wdbc.csv                # Breast cancer dataset
├── README.md               # Project documentation
└── requirements.txt        # Python requirements (optional)
```

## ▶️ How to Run

1. Clone this repository:

```
git clone https://github.com/your-username/breast-cancer-ml-app.git
cd breast-cancer-ml-app
```

2. Install dependencies:

```
pip install -r requirements.txt
```

> Or install manually:

```
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

3. Run the app:

```
streamlit run breast_cancer.py
```

4. The app will open automatically at `http://localhost:8501` in your browser.

## 📊 Supported Models

| Model                     | Supports Probabilities | ROC/PR Curve Available | Feature Importance |
|--------------------------|------------------------|------------------------|--------------------|
| Logistic Regression      | ✅                     | ✅                     | ❌                 |
| Random Forest            | ✅                     | ✅                     | ✅                 |
| Support Vector Machine   | ✅                     | ✅                     | ❌                 |
| K-Nearest Neighbors      | ✅                     | ✅                     | ❌                 |
| Gradient Boosting        | ✅                     | ✅                     | ✅                 |

## 📁 Dataset

The `wdbc.csv` file is based on the **Wisconsin Diagnostic Breast Cancer dataset**, available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

## 🤝 Contributions

Contributions are welcome! If you have ideas, improvements, or spot a bug, feel free to open an issue or submit a pull request.

## 📄 License

MIT License – you're free to use, modify, and distribute this project.
