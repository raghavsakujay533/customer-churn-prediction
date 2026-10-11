import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0b1220 0%, #101a2f 55%, #111827 100%); color: #edf2f7; }
    [data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #243047; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 20px;
        background: linear-gradient(115deg, #14243d, #172b46 58%, #123b45);
        border: 1px solid #2c4560; margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.35rem; color: #f8fafc; }
    .hero p { color: #c5d2e2; margin: .55rem 0 0 0; font-size: 1.03rem; }
    .metric-card {
        background: #121f33; border: 1px solid #2a3b54; border-radius: 15px;
        padding: 1rem 1.1rem; min-height: 105px;
    }
    .metric-label { color: #a8bad0; font-size: .9rem; }
    .metric-value { color: #f8fafc; font-size: 1.8rem; font-weight: 750; margin-top: .25rem; }
    .section-note { color: #a8bad0; font-size: .92rem; }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px; border: 1px solid #3b82f6; font-weight: 650;
    }
    .small-pill {
        display: inline-block; padding: .25rem .65rem; border-radius: 99px;
        background: #17364a; color: #8be9d4; border: 1px solid #27616b; font-size: .82rem;
    }
    h2, h3 { color: #f0f6ff; }
</style>
""", unsafe_allow_html=True)

def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner=False)
def read_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

def clean_target(series):
    # Remove rows where the target is missing and encode target labels consistently.
    valid = series.notna()
    y_raw = series[valid].astype(str).str.strip()
    labels = sorted(y_raw.unique().tolist())
    mapping = {label: idx for idx, label in enumerate(labels)}
    return valid, y_raw.map(mapping), labels, mapping

def build_pipeline(X, model_name, random_state):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ], remainder="drop")

    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=250, random_state=random_state,
            class_weight="balanced", min_samples_leaf=2
        )
    else:
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=random_state
        )
    return Pipeline([("preprocessing", prep), ("model", model)])

def format_pct(x):
    return f"{x * 100:.1f}%" if pd.notna(x) else "N/A"

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <span class="small-pill">MACHINE LEARNING • CUSTOMER ANALYTICS</span>
  <h1>📊 ChurnIQ</h1>
  <p>Explore customer behavior, train a churn classifier, review model quality, and score new customers — all in one interactive workspace.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.caption("Start by uploading a CSV dataset.")
    uploaded = st.file_uploader("Upload customer dataset (.csv)", type=["csv"])
    st.markdown("---")
    st.markdown("### What you can do")
    st.markdown("1. Preview and explore data\n2. Select the churn target\n3. Train a model\n4. Evaluate performance\n5. Download predictions")
    st.markdown("---")
    st.caption("Tip: keep customer IDs out of model features when they are only identifiers.")

if uploaded is None:
    st.info("👈 Upload your customer churn CSV from the sidebar to get started.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🔎 Explore")
        st.write("Inspect columns, missing values, and churn distribution.")
    with c2:
        st.markdown("### 🧠 Train")
        st.write("Train Logistic Regression or Random Forest with preprocessing.")
    with c3:
        st.markdown("### 📈 Evaluate")
        st.write("Review classification metrics and download scored data.")
    st.markdown("#### Expected input")
    st.write("A CSV with one row per customer and a target column such as `Churn` containing values like `Yes` / `No` or `1` / `0`.")
    st.stop()

try:
    df = read_csv(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("Please upload a CSV with at least two columns and one or more rows.")
    st.stop()

# Normalize common CSV artifacts
df.columns = [str(c).strip() for c in df.columns]
with st.sidebar:
    target_default = next((i for i, c in enumerate(df.columns) if c.lower() in ["churn", "exited", "customer_churn", "attrition"]), len(df.columns)-1)
    target = st.selectbox("Target / label column", df.columns.tolist(), index=target_default)
    model_name = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
    test_size = st.slider("Test set size", min_value=0.15, max_value=0.40, value=0.20, step=0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    excluded = st.multiselect("Exclude columns (IDs or leakage)", [c for c in df.columns if c != target])

# Overview metrics
n_rows, n_cols = df.shape
missing_count = int(df.isna().sum().sum())
unique_customers = df.shape[0]
a, b, c, d = st.columns(4)
with a: metric_card("Customer records", f"{n_rows:,}")
with b: metric_card("Available columns", f"{n_cols}")
with c: metric_card("Missing cells", f"{missing_count:,}")
with d: metric_card("Target column", target)

tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Explorer", "🧠 Train & Evaluate", "🔮 Score Customers", "ℹ️ Guide"])

with tab1:
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True, height=320)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Missing values by column")
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].head(15)
        if len(miss):
            st.bar_chart(miss)
        else:
            st.success("No missing values found.")
    with right:
        st.markdown("#### Target distribution")
        counts = df[target].fillna("Missing").astype(str).value_counts().head(20)
        st.bar_chart(counts)
    st.markdown("#### Column types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(t) for t in df.dtypes],
        "Missing values": [int(df[col].isna().sum()) for col in df.columns],
        "Unique values": [int(df[col].nunique(dropna=True)) for col in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Train your churn model")
    st.markdown('<p class="section-note">Preprocessing is fitted on the training split only to reduce data leakage. The selected target should represent the outcome you want to predict.</p>', unsafe_allow_html=True)

    if st.button("🚀 Train model", type="primary", use_container_width=False):
        valid, y_encoded, labels, mapping = clean_target(df[target])
        usable = df.loc[valid].copy()
        y = y_encoded.loc[valid].astype(int)
        feature_cols = [col for col in df.columns if col != target and col not in excluded]
        if not feature_cols:
            st.error("Choose at least one feature column.")
        elif y.nunique() < 2:
            st.error("The target column must contain at least two classes.")
        elif y.value_counts().min() < 2:
            st.error("Each target class needs at least two rows for a stratified train/test split.")
        else:
            X = usable[feature_cols].copy()
            # Convert datetimes to strings so they can be handled as categories.
            for col in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    X[col] = X[col].astype(str)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
                )
                pipeline = build_pipeline(X_train, model_name, int(random_state))
                with st.spinner("Training model and evaluating the holdout set..."):
                    pipeline.fit(X_train, y_train)
                    predictions = pipeline.predict(X_test)
                acc = accuracy_score(y_test, predictions)
                prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
                rec = recall_score(y_test, predictions, average="weighted", zero_division=0)
                f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
                st.session_state["churn_model"] = pipeline
                st.session_state["churn_features"] = feature_cols
                st.session_state["churn_labels"] = labels
                st.session_state["churn_mapping"] = mapping
                st.session_state["churn_metrics"] = (acc, prec, rec, f1)
                st.session_state["churn_eval"] = {
                    "y_test": y_test, "predictions": predictions,
                    "X_test": X_test, "model_name": model_name,
                    "target": target, "labels": labels
                }
                st.success(f"{model_name} trained successfully.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    if "churn_metrics" in st.session_state:
        acc, prec, rec, f1 = st.session_state["churn_metrics"]
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("Accuracy", format_pct(acc))
        with m2: metric_card("Weighted precision", format_pct(prec))
        with m3: metric_card("Weighted recall", format_pct(rec))
        with m4: metric_card("Weighted F1-score", format_pct(f1))

        ev = st.session_state["churn_eval"]
        st.markdown("#### Confusion matrix")
        cm = confusion_matrix(ev["y_test"], ev["predictions"], labels=list(range(len(ev["labels"]))))
        cm_df = pd.DataFrame(cm, index=[f"Actual: {x}" for x in ev["labels"]], columns=[f"Pred: {x}" for x in ev["labels"]])
        st.dataframe(cm_df, use_container_width=True)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks(range(len(ev["labels"])), ev["labels"], rotation=30, ha="right")
        ax.set_yticks(range(len(ev["labels"])), ev["labels"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        report = classification_report(
            ev["y_test"], ev["predictions"],
            labels=list(range(len(ev["labels"]))),
            target_names=[str(x) for x in ev["labels"]],
            output_dict=True, zero_division=0
        )
        st.markdown("#### Classification report")
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
        st.caption("Metrics are calculated on the held-out test split. Results depend on dataset quality, target balance, feature selection, and model settings.")

with tab3:
    st.subheader("Score customers")
    if "churn_model" not in st.session_state:
        st.info("Train a model in the **Train & Evaluate** tab first.")
    else:
        model = st.session_state["churn_model"]
        feature_cols = st.session_state["churn_features"]
        labels = st.session_state["churn_labels"]
        st.write("Upload a separate customer CSV with the same feature columns, or score the current dataset.")
        scoring_file = st.file_uploader("Optional: upload customers to score (.csv)", type=["csv"], key="score_upload")
        if scoring_file is not None:
            try:
                score_df = pd.read_csv(scoring_file)
            except Exception as exc:
                st.error(f"Could not read scoring CSV: {exc}")
                score_df = None
        else:
            score_df = df.drop(columns=[target], errors="ignore").copy()

        if score_df is not None:
            missing_features = [col for col in feature_cols if col not in score_df.columns]
            if missing_features:
                st.warning("Scoring file is missing required columns: " + ", ".join(missing_features))
            else:
                score_X = score_df[feature_cols].copy()
                try:
                    pred_codes = model.predict(score_X)
                    pred_labels = [labels[int(p)] for p in pred_codes]
                    result_df = score_df.copy()
                    result_df["Predicted_Churn"] = pred_labels
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(score_X)
                        if len(labels) == 2:
                            # Class index 1 is not guaranteed to mean "Yes"; use label mapping order and expose it.
                            positive_label = labels[1]
                            result_df[f"Probability_{positive_label}"] = probs[:, 1].round(4)
                    st.dataframe(result_df.head(100), use_container_width=True)
                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name="customer_churn_predictions.csv",
                        mime="text/csv"
                    )
                    st.caption("Predictions are model estimates, not certainties. Confirm business decisions with appropriate context.")
                except Exception as exc:
                    st.error(f"Scoring failed. Check that uploaded columns have compatible formats. Details: {exc}")

with tab4:
    st.subheader("How to use this app")
    st.markdown("""
    1. Export your dataset from the notebook as a CSV file.
    2. Upload it from the sidebar.
    3. Select the column you want to predict (often `Churn`).
    4. Exclude IDs or fields that reveal the target outcome.
    5. Open **Train & Evaluate** and click **Train model**.
    6. Review accuracy, precision, recall, F1-score, and the confusion matrix.
    7. Open **Score Customers** to predict churn for rows in the current data or a separate CSV.

    **Dataset tips**
    - One row should represent one customer.
    - Keep the target column present for training.
    - For a separate scoring CSV, include the same feature columns but do not include the target.
    - A customer ID is usually an identifier, not a useful predictive feature.
    - Accuracy alone can be misleading when churn is rare; inspect recall, precision, and the confusion matrix.

    **Privacy:** Use synthetic or appropriately authorized data. Do not upload personal or confidential customer information to a public app.
    """)
    st.markdown("#### Example: exporting a notebook DataFrame")
    st.code("""# In your Jupyter Notebook, replace df with your dataset variable
df.to_csv("customer_churn.csv", index=False)""", language="python")

st.markdown("---")
st.caption("ChurnIQ • Built with Streamlit, Pandas, Matplotlib, and Scikit-learn")
