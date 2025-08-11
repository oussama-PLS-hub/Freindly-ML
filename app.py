"""
Streamlit Guided Data Analysis & ML App
File: streamlit_data_ml_guided_app.py

How to run:
1. Create a virtual env (recommended): python -m venv venv && source venv/bin/activate (or venv\Scripts\activate on Windows)
2. Install requirements: pip install -r requirements.txt
   Example requirements.txt content:
       streamlit
       pandas
       numpy
       scikit-learn
       matplotlib
       seaborn
       joblib

3. Run:
   streamlit run streamlit_data_ml_guided_app.py

What this app does (in plain "like you're 5" language):
- Lets you pick a built-in example dataset or upload your own CSV.
- Checks the table for common problems (like missing numbers) and tells you simply what those problems mean.
- Gives you a few big friendly buttons that do the right thing for you (or you can pick another way).
- Lets you explore simple pictures of your data (histograms, boxplots, heatmaps).
- Helps you pick a question to ask (the "target").
- Lets you train a basic model (one click) and shows simple scores and explanations.
- Lets you download the trained model.

Designed to be step-by-step (baby steps). The UI explains each choice like a short, friendly sentence.

Feel free to copy this file to a GitHub repo and deploy it (Streamlit Cloud or other host).

"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib

st.set_page_config(page_title="Guided Data & ML for Everyone", layout="wide")

# ---------------------- Helpers & Explanations ----------------------

def explain_short(text_key):
    E = {
        "upload": "We need some data to play with. You can choose a demo dataset or upload your own CSV file.",
        "missing_detected": "We found empty boxes in your table. We can delete those rows, fill them with a number, or try other ways. I'll explain each choice.",
        "drop_rows": "Delete the whole row that has empty boxes. Good if only a few rows are bad.",
        "drop_cols": "Delete the whole column that has many empty boxes. Do this if the column is mostly empty.",
        "fill_mean": "Fill the empty boxes with the average number for that column. Works for numbers.",
        "fill_median": "Fill with the middle value. Safer when numbers have big outliers.",
        "fill_mode": "Fill with the most common value (like 'blue' if most rows say 'blue'). Works for categories.",
        "onehot": "Turn words into columns of 0s and 1s. Machines like numbers better.",
        "label_encode": "Turn each word into a simple number (1, 2, 3). Quick and simple.",
        "eda": "Let's draw easy pictures to understand the data: bar charts, histograms, and a correlation map.",
        "target": "Pick the column you want the computer to learn to predict. For example, 'price' or 'churn'.",
        "train": "Click the train button and the model will learn from the data. Then we'll test how good it is.",
    }
    return E.get(text_key, "")


# ---------------------- Demo datasets ----------------------

def load_demo_datasets():
    demos = {}
    iris = datasets.load_iris(as_frame=True)
    df_iris = iris.frame.copy()
    df_iris.columns = list(df_iris.columns)
    df_iris.rename(columns={"target": "target"}, inplace=True)
    demos["Iris (classification)"] = df_iris

    bc = datasets.load_breast_cancer(as_frame=True)
    df_bc = bc.frame.copy()
    df_bc = df_bc.rename(columns={"target": "target"})
    demos["Breast Cancer (classification)"] = df_bc

    diabetes = datasets.load_diabetes(as_frame=True)
    df_db = diabetes.frame.copy()
    df_db = df_db.rename(columns={"target": "target"})
    demos["Diabetes (regression)"] = df_db

    # Make a small toy 'messy' dataset so users see missing values
    df_messy = pd.DataFrame({
        "age": [25, 30, None, 22, 40, None],
        "gender": ["F", "M", "M", None, "F", "F"],
        "score": [88, 92, 85, None, 95, 78],
        "bought": [0, 1, 0, 1, 1, 0],
    })
    demos["Small messy demo"] = df_messy

    return demos


def read_uploaded_file(uploaded):
    try:
        if uploaded.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded)
        elif uploaded.name.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded)
        else:
            # try csv fallback
            uploaded.seek(0)
            return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None


# ---------------------- Session State Init ----------------------

if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "df" not in st.session_state:
    st.session_state.df = None
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "model" not in st.session_state:
    st.session_state.model = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None


# ---------------------- Stepper UI ----------------------

def goto_step(s):
    st.session_state.current_step = s


st.title("🧠 Guided Data & ML — Baby Steps for Everyone")
st.markdown("We explain each step like you're 5 years old and give big buttons to do the right thing.")

steps = [
    "Choose data", 
    "Fix missing values",
    "Types & Encoding",
    "Explore data",
    "Choose target & model",
    "Train & Evaluate",
    "Save & Download",
]

cols = st.columns(len(steps))
for i, col in enumerate(cols):
    with col:
        if i == st.session_state.current_step:
            st.button(f"➡️ {i+1}. {steps[i]}", key=f"step_{i}", on_click=goto_step, args=(i,))
        else:
            if st.button(f"{i+1}. {steps[i]}", key=f"step_btn_{i}", on_click=goto_step, args=(i,)):
                pass

st.write("---")


# ---------------------- Step 0: Upload / choose demo ----------------------

if st.session_state.current_step == 0:
    st.header("Step 1 — Choose data")
    st.write(explain_short("upload"))
    demos = load_demo_datasets()
    demo_names = list(demos.keys())

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload your CSV or Excel file (optional)", type=["csv", "xls", "xlsx"])
        if uploaded_file is not None:
            df = read_uploaded_file(uploaded_file)
            if df is not None:
                st.session_state.original_df = df.copy()
                st.session_state.df = df.copy()
                st.success("File loaded! We stored your data.")
        else:
            demo_choice = st.selectbox("Or choose a demo dataset:", demo_names)
            if st.button("Load demo dataset"):
                st.session_state.original_df = demos[demo_choice].copy()
                st.session_state.df = demos[demo_choice].copy()
                st.success(f"Loaded demo: {demo_choice}")

    with col2:
        st.markdown("**Quick preview**")
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df.head(10))
        else:
            st.info("No data yet. Upload a file or load a demo to begin.")

    st.write("\n")
    if st.session_state.df is not None:
        if st.button("Next: Fix missing values"):
            goto_step(1)


# ---------------------- Step 1: Missing values ----------------------

if st.session_state.current_step == 1:
    st.header("Step 2 — Detect & fix missing values")
    st.write(explain_short("missing_detected"))
    df = st.session_state.df
    if df is None:
        st.error("No dataset loaded. Go back to step 1 and load a dataset.")
    else:
        missing = df.isnull().sum()
        total_missing = int(missing.sum())
        st.write(f"We found **{total_missing}** empty boxes in the table.")
        if total_missing == 0:
            st.success("No missing values found — nice! You can skip or continue to other cleaning steps.")
        st.table(pd.DataFrame(missing, columns=["missing_count"]).query('missing_count > 0'))

        st.subheader("Choose what to do (click the big button)")
        colA, colB = st.columns(2)
        with colA:
            if st.button("🗑️ Delete rows with missing values"):
                st.write(explain_short("drop_rows"))
                df2 = df.dropna()
                st.session_state.df = df2
                st.success("Deleted rows with missing values. Table is updated.")
        with colB:
            if st.button("🚮 Delete columns with missing values"):
                st.write(explain_short("drop_cols"))
                col_to_drop = st.multiselect("Choose columns to drop (or leave empty to drop all with any missing):", options=list(df.columns))
                if len(col_to_drop) == 0:
                    df2 = df.dropna(axis=1)
                else:
                    df2 = df.drop(columns=col_to_drop)
                st.session_state.df = df2
                st.success("Dropped columns. Table is updated.")

        st.write("---")
        st.subheader("Fill missing values instead")
        fill_method = st.radio("Pick a fill method:", ["Mean (numbers)", "Median (numbers)", "Mode (categorical)", "Constant value", "Keep as is"])
        if fill_method != "Keep as is":
            if fill_method == "Constant value":
                const_val = st.text_input("Enter constant value (will be cast where possible):", value="0")
            if st.button("Apply fill"):
                st.write(explain_short("fill_mean") if fill_method == "Mean (numbers)" else explain_short("fill_median") if fill_method == "Median (numbers)" else explain_short("fill_mode"))
                df2 = st.session_state.df.copy()
                if fill_method == "Mean (numbers)":
                    for col in df2.select_dtypes(include=[np.number]).columns:
                        df2[col] = df2[col].fillna(df2[col].mean())
                elif fill_method == "Median (numbers)":
                    for col in df2.select_dtypes(include=[np.number]).columns:
                        df2[col] = df2[col].fillna(df2[col].median())
                elif fill_method == "Mode (categorical)":
                    for col in df2.columns:
                        if df2[col].dtype == object or df2[col].nunique() < 20:
                            try:
                                df2[col] = df2[col].fillna(df2[col].mode()[0])
                            except Exception:
                                pass
                elif fill_method == "Constant value":
                    for col in df2.columns:
                        try:
                            df2[col] = df2[col].fillna(type(df2[col].dropna().iloc[0])(const_val) if len(df2[col].dropna())>0 else const_val)
                        except Exception:
                            df2[col] = df2[col].fillna(const_val)
                st.session_state.df = df2
                st.success("Missing values filled. Table is updated.")

        st.write("\n")
        st.markdown("**Preview after cleaning**")
        st.dataframe(st.session_state.df.head(8))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back: Choose data"):
                goto_step(0)
        with col2:
            if st.button("Next: Types & Encoding"):
                goto_step(2)


# ---------------------- Step 2: Types & Encoding ----------------------

if st.session_state.current_step == 2:
    st.header("Step 3 — Data types & encoding")
    st.write("Machines like numbers. If you have words, you can convert them. Here are two easy choices:")
    st.write("1) One-hot encoding — makes a column for each word. 2) Label encoding — gives each word a number.")

    df = st.session_state.df
    if df is None:
        st.error("No dataset loaded. Go back to step 1 and load a dataset.")
    else:
        cat_cols = list(df.select_dtypes(include=[object, 'category']).columns)
        st.write(f"Detected categorical columns: {cat_cols}")
        if cat_cols:
            enc_choice = st.radio("Choose encoding:", ["One-hot (safe)", "Label encode (simple)", "Leave as is"])
            if enc_choice == "One-hot (safe)":
                if st.button("Apply one-hot encoding"):
                    st.write(explain_short("onehot"))
                    df2 = pd.get_dummies(df, columns=cat_cols, drop_first=False)
                    st.session_state.df = df2
                    st.success("One-hot encoding applied.")
            elif enc_choice == "Label encode (simple)":
                if st.button("Apply label encoding"):
                    st.write(explain_short("label_encode"))
                    df2 = df.copy()
                    for c in cat_cols:
                        try:
                            le = LabelEncoder()
                            df2[c] = le.fit_transform(df2[c].astype(str))
                        except Exception:
                            pass
                    st.session_state.df = df2
                    st.success("Label encoding applied.")
            else:
                st.info("No encoding applied.")
        else:
            st.success("No categorical columns detected.")

        st.markdown("**Preview**")
        st.dataframe(st.session_state.df.head(8))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back: Missing values"):
                goto_step(1)
        with col2:
            if st.button("Next: Explore data"):
                goto_step(3)


# ---------------------- Step 3: EDA ----------------------

if st.session_state.current_step == 3:
    st.header("Step 4 — Explore your data (pictures)")
    st.write(explain_short("eda"))
    df = st.session_state.df
    if df is None:
        st.error("No dataset loaded. Go back to step 1 and load a dataset.")
    else:
        st.subheader("Table & summary")
        st.write(df.describe(include='all'))
        st.dataframe(df.head(15))

        st.subheader("Histograms (pick a numeric column)")
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        col = st.selectbox("Numeric column for histogram:", options=numeric_cols)
        if col:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

        st.subheader("Correlation heatmap (numeric only)")
        if len(numeric_cols) >= 2:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', ax=ax2)
            st.pyplot(fig2)
        else:
            st.info("Not enough numeric columns for correlation map.")

        st.subheader("Scatter / boxplot (quick) — pick two columns")
        colx = st.selectbox("X (numeric)", options=numeric_cols, key="xcol")
        coly = st.selectbox("Y (numeric)", options=numeric_cols, key="ycol")
        if colx and coly:
            fig3, ax3 = plt.subplots()
            sns.scatterplot(x=df[colx], y=df[coly], ax=ax3)
            ax3.set_title(f"Scatter: {colx} vs {coly}")
            st.pyplot(fig3)

        st.write("\n")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back: Types & Encoding"):
                goto_step(2)
        with col2:
            if st.button("Next: Choose target & model"):
                goto_step(4)


# ---------------------- Step 4: Target & Model ----------------------

if st.session_state.current_step == 4:
    st.header("Step 5 — Pick your question (target) and a model")
    df = st.session_state.df
    if df is None:
        st.error("No dataset loaded. Go back to step 1 and load a dataset.")
    else:
        st.write(explain_short("target"))
        cols = list(df.columns)
        target = st.selectbox("Choose the target column (what you want to predict):", options=cols)
        if target:
            # Simple auto-detect problem type
            unique_vals = df[target].nunique(dropna=True)
            dtype = df[target].dtype
            if np.issubdtype(dtype, np.number) and unique_vals > 20:
                problem = "regression"
            else:
                problem = "classification"
            st.session_state.problem_type = problem
            st.info(f"Auto-detected problem type: {problem}")

            st.subheader("Choose model (or keep recommended)")
            if problem == "classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=200),
                    "Random Forest": RandomForestClassifier(n_estimators=100),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
                }

            model_choice = st.selectbox("Model:", options=list(models.keys()), index=0)
            test_size = st.slider("Test set size (percent):", min_value=10, max_value=50, value=20)
            scale_choice = st.checkbox("Standard scale numeric features?", value=True)

            if st.button("Train (one-click)"):
                st.write(explain_short("train"))
                X = df.drop(columns=[target]).copy()
                y = df[target].copy()
                # Drop rows where target is null
                mask = y.notnull()
                X = X.loc[mask]
                y = y.loc[mask]

                # Simple handling: drop non-numeric columns for now or encode them
                X_processed = X.copy()
                for c in X_processed.select_dtypes(include=[object, 'category']).columns:
                    X_processed[c] = LabelEncoder().fit_transform(X_processed[c].astype(str))

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size/100.0, random_state=42)

                # Scaling
                if scale_choice:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                model = models[model_choice]
                # Fit
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                metrics = {}
                if problem == "classification":
                    try:
                        metrics['accuracy'] = accuracy_score(y_test, y_pred)
                        metrics['precision'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
                        metrics['recall'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
                        metrics['f1'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    except Exception:
                        # fallback for binary or tricky labels
                        metrics['accuracy'] = accuracy_score(y_test, y_pred)
                else:
                    metrics['RMSE'] = mean_squared_error(y_test, y_pred, squared=False)
                    metrics['MAE'] = mean_absolute_error(y_test, y_pred)
                    metrics['R2'] = r2_score(y_test, y_pred)

                st.session_state.model = model
                st.session_state.trained = True
                st.session_state.metrics = metrics
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.selected_target = target
                st.session_state.feature_names = list(X_processed.columns)
                st.success("Training finished — check results in the next step!")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back: Explore data"):
                goto_step(3)
        with col2:
            if st.button("Next: Train & Evaluate"):
                goto_step(5)


# ---------------------- Step 5: Train & Evaluate ----------------------

if st.session_state.current_step == 5:
    st.header("Step 6 — Results & evaluation")
    if not st.session_state.trained:
        st.warning("You haven't trained a model yet. Go back to the previous step and click Train.")
    else:
        st.success("We trained a model for you!")
        st.write("**Metrics:**")
        for k, v in st.session_state.metrics.items():
            st.write(f"- **{k}**: {round(v, 3)}")

        if st.session_state.problem_type == "classification":
            st.subheader("Confusion matrix")
            y_test = st.session_state.y_test
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

            # Feature importance if available
            if hasattr(st.session_state.model, 'feature_importances_'):
                st.subheader("Feature importance")
                importances = st.session_state.model.feature_importances_
                fi = pd.Series(importances, index=st.session_state.feature_names).sort_values(ascending=False)
                st.bar_chart(fi.head(10))
        else:
            st.subheader("Predictions vs True (first 30)")
            y_test = st.session_state.y_test
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            df_results = pd.DataFrame({"true": y_test, "pred": y_pred}).reset_index(drop=True)
            st.dataframe(df_results.head(30))

        st.write("\n")
        col1, col2 = st.columns([1,2])
        with col1:
            if st.button("Back: Choose model"):
                goto_step(4)
        with col2:
            if st.button("Next: Save & Download"):
                goto_step(6)


# ---------------------- Step 6: Save & Download ----------------------

if st.session_state.current_step == 6:
    st.header("Step 7 — Save your model")
    if not st.session_state.trained:
        st.error("No trained model available. Train before saving.")
    else:
        st.write("You can download the trained model (.joblib) and the names of the features. You can later load it in Python with joblib.load().")
        buffer = io.BytesIO()
        joblib.dump(st.session_state.model, buffer)
        buffer.seek(0)
        st.download_button("Download model (.joblib)", data=buffer, file_name="model.joblib")

        # Also let user download a small metadata JSON / CSV with feature names and target
        meta = {
            "target": st.session_state.selected_target,
            "problem_type": st.session_state.problem_type,
            "features": st.session_state.feature_names,
            "metrics": st.session_state.metrics,
        }
        st.json(meta)

        if st.button("Start again with original data"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.trained = False
            st.session_state.model = None
            st.session_state.metrics = {}
            st.success("Reset to original dataset.")

        if st.button("Back: Results"):
            goto_step(5)


# ---------------------- Footer / Tips ----------------------

st.write("---")
st.caption("Tip: This app is a friendly, opinionated starter. For production use you should add proper preprocessing, cross-validation, hyperparameter tuning, error handling, and domain validation.")


# End of file
