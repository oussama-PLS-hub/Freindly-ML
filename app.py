# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

st.set_page_config(page_title="Guided Data Science Studio", layout="wide")

# --- Steps labels (CRISP-DM inspired) ---
STEPS = [
    "1. Business goal",
    "2. Data & context",
    "3. Clean & prep",
    "4. Model & train",
    "5. Evaluate & explain",
    "6. Save & deploy"
]

# --- Session state init ---
def init_state():
    if "df_original" not in st.session_state:
        st.session_state.df_original = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "model" not in st.session_state:
        st.session_state.model = None
    if "trained" not in st.session_state:
        st.session_state.trained = False
    if "problem" not in st.session_state:
        st.session_state.problem = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    if "feature_names" not in st.session_state:
        st.session_state.feature_names = []

init_state()

# --- Small helper: demo datasets ---
def load_demo(name):
    if name == "Iris":
        from sklearn import datasets
        iris = datasets.load_iris(as_frame=True)
        return iris.frame.copy()
    if name == "Titanic (clean)":
        try:
            return sns.load_dataset("titanic")
        except Exception:
            return pd.DataFrame({"note": ["seaborn dataset not available"]})
    if name == "Small messy":
        return pd.DataFrame({
            "age": [25, 30, np.nan, 22, 40, None],
            "gender": ["F", "M", "M", None, "F", "F"],
            "score": [88, 92, 85, None, 95, 78],
            "bought": [0, 1, 0, 1, 1, 0]
        })
    return pd.DataFrame()

# --- Small CSS to make the UI prettier ---
st.markdown("""
<style>
.big-header{
  border-radius:12px;
  padding:18px;
  background: linear-gradient(90deg,#4b6cb7,#182848);
  color: white;
}
.step-card{
  border-radius:10px;
  padding:12px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.12);
  margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar: stepper, beginner toggle, reset ---
with st.sidebar:
    st.markdown("<div class='big-header'><h2>Guided Data Science Studio</h2><div style='font-size:14px'>CRISP-DM inspired • Beginner friendly</div></div>", unsafe_allow_html=True)
    st.markdown("## Steps")
    for i, s in enumerate(STEPS):
        if i == st.session_state.step:
            st.markdown(f"<div class='step-card' style='background:#eef3ff'><strong>{s}</strong></div>", unsafe_allow_html=True)
        else:
            if st.button(s, key=f"jump_{i}"):
                st.session_state.step = i
    st.markdown("---")
    st.checkbox("Beginner mode (explain like I'm 5)", value=True, key="beginner")
    st.button("Reset app",
              key="reset",
              on_click=lambda: st.session_state.update({
                  'df_original': None,
                  'df': None,
                  'step': 0,
                  'model': None,
                  'trained': False,
                  'problem': None,
                  'X_test': None,
                  'y_test': None,
                  'feature_names': []
              }))

st.title("🧭 Guided Data Science Studio — CRISP-DM for everyone")
st.write("We follow a simple, proven process (CRISP-DM). We'll explain each step in plain words and give big friendly buttons to act.")

# -------------------------
# Step 1 — Business goal
# -------------------------
def step_business():
    st.header("Step 1 — Business goal (Why are we doing this?)")
    st.write("Start with the question you want to answer. This helps pick the right data and models.")
    st.info("Examples: `Predict whether a customer will buy (yes/no)`, `Estimate house price (number)`")
    goal = st.text_area("Write your objective in one sentence (short & clear):", value=st.session_state.get('project_goal', ''))
    if st.button("Save goal"):
        st.session_state.project_goal = goal
        st.success("Saved ✅")
    st.markdown("**Next** — Load data in the next step (demo or your file).")
    if st.button("Next: Data & context"):
        st.session_state.step = 1

# -------------------------
# Step 2 — Data & context
# -------------------------
def step_data():
    st.header("Step 2 — Data & context (Load and look at the table)")
    st.write("Upload your CSV/XLSX or try a demo dataset to experiment.")
    demo = st.selectbox("Choose demo dataset", ["Iris", "Titanic (clean)", "Small messy"])
    if st.button("Load demo"):
        st.session_state.df_original = load_demo(demo)
        st.session_state.df = st.session_state.df_original.copy()
        st.success("Demo loaded")
    uploaded = st.file_uploader("Or upload CSV / Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            if str(uploaded.name).lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df_original = df.copy()
            st.session_state.df = df.copy()
            st.success("File loaded!")
        except Exception as e:
            st.error("Could not read file: " + str(e))

    df = st.session_state.df
    if df is None:
        st.info("No data loaded yet.")
    else:
        st.subheader("Quick overview")
        st.write(f"Rows: {df.shape[0]}  •  Columns: {df.shape[1]}")
        st.dataframe(df.head(8))
        st.markdown("**Columns & types**")
        st.table(pd.DataFrame({ "column": df.columns, "dtype": df.dtypes.astype(str) }))
        if st.button("Next: Clean & prepare"):
            st.session_state.step = 2

# -------------------------
# Step 3 — Clean & prepare
# -------------------------
def step_clean():
    st.header("Step 3 — Clean & prepare (Make the table ready)")
    st.write("Fix missing values, turn words into numbers, and keep the useful columns.")

    df = st.session_state.df
    if df is None:
        st.error("No data loaded. Go back to Step 2.")
        return

    st.subheader("Missing values (empty boxes)")
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.success("No missing values detected.")
    else:
        st.table(miss)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Drop rows with any missing"):
                st.session_state.df = df.dropna()
                st.success("Rows dropped")
        with col2:
            if st.button("Drop columns with >50% missing"):
                thresh = int(0.5 * len(df))
                st.session_state.df = df.dropna(axis=1, thresh=thresh)
                st.success("Columns dropped")
        with col3:
            if st.button("Fill with mean (numbers) & mode (words)"):
                df2 = df.copy()
                for c in df2.columns:
                    if pd.api.types.is_numeric_dtype(df2[c]):
                        df2[c] = df2[c].fillna(df2[c].mean())
                    else:
                        if not df2[c].mode().empty:
                            df2[c] = df2[c].fillna(df2[c].mode().iloc[0])
                        else:
                            df2[c] = df2[c].fillna('')
                st.session_state.df = df2
                st.success("Imputation applied")

    st.subheader("Convert words to numbers (encoding)")
    cat_cols = [c for c in st.session_state.df.columns if st.session_state.df[c].dtype == object or st.session_state.df[c].dtype.name == 'category']
    st.write("Categorical columns:", cat_cols)
    if cat_cols:
        if st.button("Auto-encode categories (one-hot small / label large)"):
            df2 = st.session_state.df.copy()
            for c in cat_cols:
                try:
                    if df2[c].nunique() <= 8:
                        df2 = pd.get_dummies(df2, columns=[c], prefix=c, drop_first=False)
                    else:
                        le = LabelEncoder()
                        df2[c] = le.fit_transform(df2[c].astype(str))
                except Exception:
                    pass
            st.session_state.df = df2
            st.success("Encoding applied")

    st.subheader("Feature selection (optional)")
    cols = st.multiselect("Choose columns to KEEP (empty = keep all)", options=list(st.session_state.df.columns))
    if cols:
        st.session_state.df = st.session_state.df[cols]
        st.success("Filtered columns")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back: Data & context"):
            st.session_state.step = 1
    with col2:
        if st.button("Next: Model & train"):
            st.session_state.step = 3

# helper: detect classification vs regression
def detect_problem_type(series):
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() > 20:
            return "regression"
        else:
            return "classification"
    else:
        return "classification"

# -------------------------
# Step 4 — Model & train
# -------------------------
def step_model():
    st.header("Step 4 — Model & train (Teach the computer)")
    df = st.session_state.df
    if df is None:
        st.error("No data loaded.")
        return

    st.subheader("Choose the target (what we want to predict)")
    target = st.selectbox("Target column", options=list(df.columns))
    st.write("If the target has many different numbers → regression. If the target is categories → classification.")
    if target:
        problem = detect_problem_type(df[target])
        st.write(f"Auto-detected problem type: **{problem}**")
        st.session_state.problem = problem
        features = [c for c in df.columns if c != target]
        st.write("Features (inputs):", features)

        if problem == "classification":
            model_choice = st.selectbox("Model", ["Random Forest (recommended)", "Logistic Regression"])
        else:
            model_choice = st.selectbox("Model", ["Random Forest Regressor (recommended)", "Linear Regression"])

        test_size = st.slider("Test set size (%)", 10, 40, 20)
        scale = st.checkbox("Scale numeric features (recommended)", value=True)

        if st.button("Train model"):
            X = df[features].copy()
            y = df[target].copy()

            # Encode remaining text columns
            for c in X.select_dtypes(include=['object', 'category']):
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

            # drop rows with missing target
            mask = y.notnull()
            X = X.loc[mask]
            y = y.loc[mask]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)

            if scale:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if "Random Forest" in model_choice:
                if problem == "classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif "Logistic" in model_choice:
                model = LogisticRegression(max_iter=400)
            else:
                model = LinearRegression()

            model.fit(X_train, y_train)
            st.session_state.model = model
            st.session_state.trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.feature_names = features
            st.success("Model trained ✅")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back: Clean & prepare"):
            st.session_state.step = 2
    with col2:
        if st.button("Next: Evaluate & explain"):
            st.session_state.step = 4

# -------------------------
# Step 5 — Evaluate & explain
# -------------------------
def step_evaluate():
    st.header("Step 5 — Evaluate & explain (How good is the model?)")
    if not st.session_state.trained:
        st.warning("No trained model found. Train a model in the previous step.")
        return

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    problem = st.session_state.problem

    st.subheader("Metrics")
    y_pred = model.predict(X_test)

    if problem == "classification":
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        st.write(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
        st.write(f"F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"RMSE: {rmse:.3f}")
        st.write(f"MAE: {mae:.3f}")
        st.write(f"R²: {r2:.3f}")
        st.subheader("Predictions (first 20 rows)")
        st.dataframe(pd.DataFrame({'true': y_test.reset_index(drop=True), 'pred': np.round(y_pred, 3)}).head(20))

    st.markdown("---")
    if st.button("Back: Model & train"):
        st.session_state.step = 3
    if st.button("Next: Save & deploy"):
        st.session_state.step = 5

# -------------------------
# Step 6 — Save & deploy
# -------------------------
def step_save():
    st.header("Step 6 — Save & deploy (Download model & use it)")
    if not st.session_state.trained:
        st.warning("No trained model to save.")
        return

    buf = io.BytesIO()
    joblib.dump(st.session_state.model, buf)
    buf.seek(0)
    st.download_button("Download model (.joblib)", data=buf, file_name="model.joblib")

    st.markdown("**Quick usage snippet**")
    st.code("""
import joblib
model = joblib.load('model.joblib')
preds = model.predict(X_new)
    """)

    if st.button("Start a new project (reset data)"):
        st.session_state.df = st.session_state.df_original.copy() if st.session_state.df_original is not None else None
        st.session_state.model = None
        st.session_state.trained = False
        st.session_state.step = 0

# --- Router ---
if st.session_state.step == 0:
    step_business()
elif st.session_state.step == 1:
    step_data()
elif st.session_state.step == 2:
    step_clean()
elif st.session_state.step == 3:
    step_model()
elif st.session_state.step == 4:
    step_evaluate()
elif st.session_state.step == 5:
    step_save()
