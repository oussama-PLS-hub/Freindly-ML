# app.py
"""
Guided Data Science Studio — enhanced with EDA, graphs, storytelling and clear WHAT/WHY for each step.
Drop this file into your repo as `app.py`. Make sure requirements.txt includes seaborn & matplotlib.
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, mean_squared_error, mean_absolute_error, r2_score)

# --- Page config ---
st.set_page_config(page_title="Guided Data Science Studio", layout="wide", initial_sidebar_state="expanded")

# --- Small CSS for nicer UI ---
st.markdown(
    """
    <style>
    .header { padding: 14px 18px; border-radius: 12px; background: linear-gradient(90deg,#5b8def,#2a3e99); color: white;}
    .card { padding: 12px; border-radius: 8px; box-shadow: 0 2px 8px rgba(20,20,20,0.06); background: white; margin-bottom: 10px;}
    .small-muted { color: #6c6c6c; font-size: 13px; }
    </style>
    """, unsafe_allow_html=True)

# --- Step descriptions (WHAT & WHY) ---
STEP_DESCRIPTIONS = {
    0: {
        "title": "1 • Business goal",
        "what": "We write down the question we want to answer. This is the 'why' that guides everything else.",
        "why": "If we don't know the goal, we might clean the wrong data or build the wrong model. A clear goal keeps work focused."
    },
    1: {
        "title": "2 • Data & Context",
        "what": "We load the dataset and look at the table to understand columns and types.",
        "why": "Knowing what each column means helps decide which features to use and which cleaning steps are needed."
    },
    2: {
        "title": "3 • Explore & Story (EDA)",
        "what": "We draw graphs and automatically explain what they show in simple words.",
        "why": "Pictures help us see important patterns (gaps, groups, outliers, relationships) before we build models."
    },
    3: {
        "title": "4 • Clean & Prepare",
        "what": "We fix missing values, convert words to numbers, and keep useful columns.",
        "why": "Models need clean numbers. Bad or missing data makes predictions wrong — cleaning improves accuracy."
    },
    4: {
        "title": "5 • Model & Train",
        "what": "We pick a target to predict and train a simple model with one click.",
        "why": "This is where the computer learns patterns so it can make predictions on new data."
    },
    5: {
        "title": "6 • Evaluate & Explain",
        "what": "We check how well the model performs and explain the results in plain language.",
        "why": "Evaluation prevents false confidence — it shows whether the model is useful or needs improvements."
    },
    6: {
        "title": "7 • Save & Deploy",
        "what": "We export the trained model and show a code snippet to use it.",
        "why": "Saving lets you reuse the model later or deploy it to make predictions automatically."
    }
}

# --- Session state init ---
def init_state():
    s = st.session_state
    s.setdefault("step", 0)
    s.setdefault("df_original", None)
    s.setdefault("df", None)
    s.setdefault("project_goal", "")
    s.setdefault("problem", None)
    s.setdefault("model", None)
    s.setdefault("trained", False)
    s.setdefault("X_test", None)
    s.setdefault("y_test", None)
    s.setdefault("feature_names", [])
    s.setdefault("beginner", True)

init_state()

# --- Sidebar: stepper & settings ---
with st.sidebar:
    st.markdown(f"<div class='header'><h3 style='margin:0'>Guided Data Science Studio</h3><div class='small-muted'>CRISP-DM inspired — beginner friendly</div></div>", unsafe_allow_html=True)
    st.markdown("## Steps")
    for i in range(7):
        title = STEP_DESCRIPTIONS[i]["title"]
        if st.button(title, key=f"step_{i}"):
            st.session_state.step = i
    st.markdown("---")
    st.session_state.beginner = st.checkbox("Beginner mode (explain simply)", value=True)
    st.markdown("**Tips:** Upload a CSV, try the demo, and follow the big buttons.")

# --- Utility helpers for human-friendly explanations ---


def interpret_numeric(series: pd.Series, name: str, beginner=True) -> str:
    """Return a human-friendly short interpretation of a numeric column."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return f"Column **{name}** has no values (all missing)."
    mean = s.mean()
    med = s.median()
    std = s.std()
    skew = s.skew()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low_out = (s < (q1 - 1.5 * iqr)).sum() if iqr > 0 else 0
    high_out = (s > (q3 + 1.5 * iqr)).sum() if iqr > 0 else 0
    miss_pct = 100 * (series.isnull().sum() / len(series))

    # beginner-friendly phrasing
    if beginner:
        lines = []
        lines.append(f"**{name}** has *{n}* real numbers (missing: {miss_pct:.1f}%).")
        lines.append(f"Most values are around **{med:.2f}** (middle) and the average is **{mean:.2f}**.")
        if abs(skew) < 0.5:
            lines.append("The numbers are fairly balanced (not lopsided).")
        elif skew > 0:
            lines.append("There is a long tail to the right — a few much bigger values push the average up.")
        else:
            lines.append("There is a long tail to the left — some much smaller values pull the average down.")
        if low_out + high_out > 0:
            lines.append(f"There are about **{low_out + high_out}** possible outliers (very small or very big values).")
        lines.append(f"Typical spread: middle 50% values go from **{q1:.2f}** to **{q3:.2f}**.")
        return " ".join(lines)
    else:
        return (f"{name}: n={n}, mean={mean:.3f}, median={med:.3f}, std={std:.3f}, "
                f"skew={skew:.3f}, missing%={miss_pct:.2f}, outliers_low={low_out}, outliers_high={high_out}")


def interpret_categorical(series: pd.Series, name: str, beginner=True) -> str:
    """Return interpretation of categorical column."""
    n = len(series.dropna())
    miss_pct = 100 * (series.isnull().sum() / len(series))
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return f"Column **{name}** has no values (all missing)."
    top = counts.index[0]
    top_pct = 100 * counts.iloc[0] / counts.sum()
    unique = series.nunique(dropna=True)
    if beginner:
        lines = []
        lines.append(f"**{name}** has **{unique}** different values (missing: {miss_pct:.1f}%).")
        lines.append(f"The most common one is **{top}** (about {top_pct:.0f}% of rows).")
        if unique > 12:
            lines.append("This column has many different values (high cardinality).")
        return " ".join(lines)
    else:
        return f"{name}: unique={unique}, top={top} ({top_pct:.1f}%), missing%={miss_pct:.2f}"


def correlation_insights(df_numeric: pd.DataFrame, threshold=0.6) -> (list, str):
    """Return list of (colA, colB, corr) pairs with absolute corr > threshold and a textual summary."""
    corr = df_numeric.corr().abs()
    pairs = []
    n = df_numeric.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            c = corr.iloc[i, j]
            if pd.notnull(c) and c >= threshold:
                pairs.append((corr.index[i], corr.columns[j], float(c)))
    if not pairs:
        return [], "No strong correlations detected (no absolute correlation >= {:.2f}).".format(threshold)
    # build short story
    lines = []
    for a, b, v in sorted(pairs, key=lambda x: -x[2])[:6]:
        lines.append(f"**{a}** and **{b}** are strongly related (corr ≈ {v:.2f}).")
    return pairs, " ".join(lines)


def summarize_dataset(df: pd.DataFrame, target=None, beginner=True) -> str:
    """Return a 4-6 sentence summary story about the dataset (auto-generated)."""
    if df is None:
        return "No data loaded."
    rows, cols = df.shape
    missing_total = int(df.isnull().sum().sum())
    numeric = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(exclude=[np.number])
    story = []
    story.append(f"This table has **{rows}** rows and **{cols}** columns. There are **{missing_total}** missing values total.")
    if not numeric.empty:
        # largest skew numeric
        skews = numeric.skew().abs().sort_values(ascending=False)
        if not skews.empty:
            top_skew = skews.index[0]
            story.append(f"The column **{top_skew}** looks skewed — that means many values cluster but some are far away.")
    if not cat.empty:
        top_card = cat.nunique().sort_values(ascending=False).index[0]
        story.append(f"Categorical column **{top_card}** has many different values.")
    if target is not None and target in df.columns:
        if pd.api.types.is_numeric_dtype(df[target]):
            # top correlated features
            if numeric.shape[1] > 1:
                corrs = numeric.corrwith(df[target]).abs().sort_values(ascending=False)
                corrs = corrs.drop(target, errors='ignore')
                if not corrs.empty:
                    top = corrs.index[0]
                    story.append(f"The feature **{top}** is the most correlated with the target **{target}** (useful for prediction).")
    # final quick recommendation
    story.append("Next: look at the pictures (histograms and heatmap) to see outliers and relationships, then clean missing values.")
    return " ".join(story)


# --- Demo loader ---
def load_demo(name: str):
    if name == "Iris":
        from sklearn import datasets
        iris = datasets.load_iris(as_frame=True)
        return iris.frame.copy()
    if name == "Small messy":
        return pd.DataFrame({
            "age": [25, 30, np.nan, 22, 40, None, 35, 28, 120],  # one extreme
            "gender": ["F", "M", "M", None, "F", "F", "M", "F", "M"],
            "score": [88, 92, 85, None, 95, 78, 90, 82, 45],
            "bought": [0, 1, 0, 1, 1, 0, 1, 0, 0]
        })
    # fallback: small random dataset
    df = pd.DataFrame({
        "A": np.random.normal(10, 2, 200),
        "B": np.random.exponential(1.5, 200),
        "C": np.random.randint(0, 3, 200)
    })
    return df


# ---------------------------
# UI: header & current step
# ---------------------------
st.markdown(f"<div class='card'><h2 style='margin:6px 0'>{STEP_DESCRIPTIONS[st.session_state.step]['title']}</h2>"
            f"<div class='small-muted'>{STEP_DESCRIPTIONS[st.session_state.step]['what']}</div>"
            f"<div style='height:6px'></div><div class='small-muted'><strong>Why:</strong> {STEP_DESCRIPTIONS[st.session_state.step]['why']}</div></div>",
            unsafe_allow_html=True)

# ---------------------------
# Step 0 - Business goal
# ---------------------------
def step_business():
    st.subheader("Write the goal (one short sentence)")
    st.write("Example: 'Predict whether a customer will buy (yes/no)' or 'Estimate house price (number)'.")
    goal = st.text_area("Project goal:", value=st.session_state.project_goal, height=80)
    if st.button("Save goal"):
        st.session_state.project_goal = goal
        st.success("Goal saved ✅")
    if st.button("Next: Data & Context"):
        st.session_state.step = 1

# ---------------------------
# Step 1 - Data & context
# ---------------------------
def step_data():
    st.subheader("Load data (demo or upload your file)")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        demo_choice = st.selectbox("Or choose a demo dataset", ["Iris", "Small messy"])
        if st.button("Load demo"):
            st.session_state.df_original = load_demo(demo_choice)
            st.session_state.df = st.session_state.df_original.copy()
            st.success(f"Demo '{demo_choice}' loaded")
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                st.session_state.df_original = df.copy()
                st.session_state.df = df.copy()
                st.success("File loaded!")
            except Exception as e:
                st.error("Could not read file: " + str(e))
    with col2:
        st.markdown("**Quick tips**")
        st.write("- CSV format is best. Excel is supported too.")
        st.write("- First row should contain column names.")
    df = st.session_state.df
    if df is not None:
        st.markdown("**Preview**")
        st.dataframe(df.head(8))
        st.markdown("**Columns & types**")
        dtypes = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
        st.table(dtypes)
        if st.button("Next: Explore & Story"):
            st.session_state.step = 2
    else:
        st.info("Load data to continue.")

# ---------------------------
# Step 2 - Explore & Story (EDA)
# ---------------------------
def step_eda():
    st.subheader("Explore the data — graphs and auto-story")
    df = st.session_state.df
    if df is None:
        st.error("No data loaded. Go back to Step 2.")
        return

    # Top summary story
    st.markdown("### Quick dataset story")
    target_for_story = st.selectbox("Optionally select a target (for story):", options=[None] + list(df.columns))
    story = summarize_dataset(df, target_for_story, beginner=st.session_state.beginner)
    st.info(story)

    # Missing values overview
    st.markdown("### Missing values")
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.success("No missing values detected.")
    else:
        st.table(miss)

    # Numeric / Categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    st.markdown(f"**Numeric columns**: {num_cols}")
    st.markdown(f"**Categorical columns**: {cat_cols}")

    # Interactive: pick a column to visualize
    st.markdown("---")
    st.write("#### Column explorer")
    col_to_plot = st.selectbox("Pick a column to visualize and interpret:", options=list(df.columns))
    if col_to_plot:
        series = df[col_to_plot]
        if pd.api.types.is_numeric_dtype(series):
            # show histogram + boxplot
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(series.dropna(), kde=True, ax=axes[0])
            axes[0].set_title(f"Histogram — {col_to_plot}")
            sns.boxplot(x=series.dropna(), ax=axes[1])
            axes[1].set_title("Boxplot (outliers)")
            st.pyplot(fig)
            text = interpret_numeric(series, col_to_plot, beginner=st.session_state.beginner)
            st.markdown(f"**Interpretation:** {text}")
        else:
            # categorical countplot
            fig, ax = plt.subplots(figsize=(8, 3))
            order = series.value_counts().index[:20]  # top 20
            sns.countplot(y=series, order=order, ax=ax)
            ax.set_title(f"Counts — {col_to_plot}")
            st.pyplot(fig)
            text = interpret_categorical(series, col_to_plot, beginner=st.session_state.beginner)
            st.markdown(f"**Interpretation:** {text}")

    # Correlation heatmap for numeric columns
    st.markdown("---")
    st.write("### Correlations (numeric columns)")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        st.pyplot(fig)
        pairs, corr_text = correlation_insights(df[num_cols], threshold=0.6)
        st.markdown("**Auto-insights:**")
        st.write(corr_text)
    else:
        st.info("Not enough numeric columns to show correlations.")

    # Pairplot if small and numeric count limited
    if len(num_cols) > 1 and df.shape[0] <= 500 and len(num_cols) <= 6:
        st.markdown("### Pairplot (scatter matrix)")
        pair_fig = sns.pairplot(df[num_cols].dropna().sample(min(300, len(df))))
        st.pyplot(pair_fig.fig)
    else:
        st.info("Pairplot skipped (dataset too large or too many numeric columns).")

    # Quick target correlation ranking if user selects a target
    st.markdown("---")
    st.write("### Which features are most related to the target?")
    target_col = st.selectbox("Pick a target to check feature correlations (optional):", options=[None] + list(df.columns))
    if target_col:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            numeric = df.select_dtypes(include=[np.number])
            if target_col in numeric.columns:
                corrs = numeric.corrwith(df[target_col]).abs().sort_values(ascending=False).drop(target_col, errors='ignore')
                top_corrs = corrs.head(6)
                st.write("Top numeric features correlated with target:")
                st.table(pd.DataFrame({"feature": top_corrs.index, "abs_corr": top_corrs.values}))
                if st.session_state.beginner:
                    if not top_corrs.empty:
                        top_feat = top_corrs.index[0]
                        st.info(f"The feature **{top_feat}** is most related to **{target_col}**, so it is especially useful when trying to predict the target.")
        else:
            # For categorical target, show group means for numeric features
            st.write("Target is categorical. We show average values of numeric features per category:")
            if df.select_dtypes(include=[np.number]).shape[1] >= 1:
                st.dataframe(df.groupby(target_col)[df.select_dtypes(include=[np.number]).columns].mean().round(3))
            else:
                st.info("No numeric features to show group means.")

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back: Data & Context"):
            st.session_state.step = 1
    with col2:
        if st.button("Next: Clean & Prepare"):
            st.session_state.step = 3
    with col3:
        if st.button("Save snapshot (download CSV)"):
            buf = io.StringIO()
            st.session_state.df.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("Download cleaned snapshot", data=buf, file_name="data_snapshot.csv", mime="text/csv")


# ---------------------------
# Step 3 - Clean & Prepare
# ---------------------------
def step_clean():
    st.subheader("Clean & Prepare — make your table model-ready")
    df = st.session_state.df
    if df is None:
        st.error("No data loaded. Go back to Step 2.")
        return

    # show missing values again
    st.write("Missing values by column:")
    miss = df.isnull().sum().sort_values(ascending=False)
    st.table(miss[miss > 0])

    st.write("Choose a cleaning action (big buttons):")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Drop rows with missing"):
            st.session_state.df = df.dropna()
            st.success("Rows with missing values dropped.")
    with c2:
        if st.button("Fill numbers with mean, words with mode"):
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
            st.success("Imputation applied.")
    with c3:
        if st.button("Drop columns with >50% missing"):
            thresh = int(0.5 * len(df))
            st.session_state.df = df.dropna(axis=1, thresh=thresh)
            st.success("Columns dropped.")

    st.markdown("---")
    # encoding options
    st.write("Encoding: convert words to numbers (machines prefer numbers).")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    st.write(f"Detected categorical columns: {cat_cols}")
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
                except Exception as e:
                    st.warning(f"Could not encode {c}: {e}")
            st.session_state.df = df2
            st.success("Encoding applied.")
    else:
        st.info("No categorical columns to encode.")

    # Feature filter
    st.markdown("---")
    st.write("Optional: select columns to keep")
    chosen = st.multiselect("Choose columns to keep (leave empty to keep all):", options=list(st.session_state.df.columns))
    if chosen:
        st.session_state.df = st.session_state.df[chosen]
        st.success("Columns filtered.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back: Explore & Story"):
            st.session_state.step = 2
    with col2:
        if st.button("Next: Model & Train"):
            st.session_state.step = 4

# ---------------------------
# Step 4 - Model & Train
# ---------------------------
def detect_problem_type(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) > 20:
            return "regression"
        else:
            return "classification"
    else:
        return "classification"


def step_model():
    st.subheader("Model & Train — teach the computer")
    df = st.session_state.df
    if df is None:
        st.error("No data loaded.")
        return

    st.write("Choose the target column (what we want to predict):")
    target = st.selectbox("Target column", options=list(df.columns))
    if not target:
        st.info("Pick a target to proceed.")
        return

    problem = detect_problem_type(df[target])
    st.write(f"Auto-detected problem type: **{problem}**")
    st.session_state.problem = problem

    features = [c for c in df.columns if c != target]
    st.write("Features (inputs) will be:", features)
    test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20)
    scale = st.checkbox("Scale numeric features (recommended)", value=True)

    model_choice = None
    if problem == "classification":
        model_choice = st.selectbox("Choose model", ["Random Forest (recommended)", "Logistic Regression"])
    else:
        model_choice = st.selectbox("Choose model", ["Random Forest Regressor (recommended)", "Linear Regression"])

    if st.button("Train model (one click)"):
        X = df[features].copy()
        y = df[target].copy()
        # encode text features here
        for c in X.select_dtypes(include=['object', 'category']).columns:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        # drop rows with missing target
        mask = y.notnull()
        X = X.loc[mask]
        y = y.loc[mask]
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100.0, random_state=42)
        # scale numeric features if requested
        scaler = None
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        # model select
        if "Random Forest" in model_choice:
            if problem == "classification":
                model = RandomForestClassifier(n_estimators=150, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=150, random_state=42)
        elif "Logistic" in model_choice:
            model = LogisticRegression(max_iter=400)
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)
        # save state
        st.session_state.model = model
        st.session_state.trained = True
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.feature_names = features
        st.session_state.scaler = scaler
        st.success("Model trained ✅")
        # show quick feature importance if possible
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.markdown("**Top features (from Random Forest importance):**")
            st.write(fi.head(8).round(3))
            st.bar_chart(fi.head(8))
        elif hasattr(model, "coef_"):
            try:
                coefs = pd.Series(model.coef_.ravel(), index=features).abs().sort_values(ascending=False)
                st.markdown("**Feature coefficients (magnitude):**")
                st.write(coefs.head(8).round(3))
            except Exception:
                pass

    st.markdown("---")
    if st.button("Back: Clean & Prepare"):
        st.session_state.step = 3
    if st.button("Next: Evaluate & Explain"):
        st.session_state.step = 5

# ---------------------------
# Step 5 - Evaluate & Explain
# ---------------------------
def step_evaluate():
    st.subheader("Evaluate & Explain — how good is our model?")
    if not st.session_state.trained:
        st.warning("No trained model found. Train a model first.")
        return
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    problem = st.session_state.problem

    y_pred = model.predict(X_test)
    if problem == "classification":
        st.write("**Classification metrics**")
        st.write(f"- Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        st.write(f"- Precision (macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
        st.write(f"- Recall (macro): {recall_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
        st.write(f"- F1 (macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
        # plain-English explanation of results
        st.markdown("**Interpretation (plain language):**")
        acc = accuracy_score(y_test, y_pred)
        if acc > 0.85:
            st.success(f"The model looks pretty good (accuracy {acc:.2%}). It's likely useful for the task.")
        elif acc > 0.6:
            st.info(f"The model is okay (accuracy {acc:.2%}) but could be improved with more data or tuning.")
        else:
            st.error(f"The model is not very accurate yet (accuracy {acc:.2%}). Try cleaning data more, adding features, or using a different model.")
    else:
        st.write("**Regression metrics**")
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"- RMSE: {rmse:.3f}")
        st.write(f"- MAE: {mae:.3f}")
        st.write(f"- R²: {r2:.3f}")
        st.markdown("**Predictions vs True (first 30)**")
        comp_df = pd.DataFrame({"true": y_test.reset_index(drop=True), "pred": np.round(y_pred, 3)})
        st.dataframe(comp_df.head(30))
        # interpretation
        st.markdown("**Interpretation (plain language):**")
        if r2 > 0.7:
            st.success(f"The model explains a lot of the variation (R²={r2:.2f}). Good job!")
        elif r2 > 0.4:
            st.info(f"The model explains some variation (R²={r2:.2f}). Consider improvements.")
        else:
            st.error(f"The model is weak (R²={r2:.2f}). Try improving features or model.")

    # Navigation
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back: Model & Train"):
            st.session_state.step = 4
    with c2:
        if st.button("Next: Save & Deploy"):
            st.session_state.step = 6

# ---------------------------
# Step 6 - Save & Deploy
# ---------------------------
def step_save():
    st.subheader("Save & Deploy")
    if not st.session_state.trained:
        st.warning("No trained model to save yet.")
        return
    buf = io.BytesIO()
    joblib.dump(st.session_state.model, buf)
    buf.seek(0)
    st.download_button("Download model (.joblib)", data=buf, file_name="model.joblib")
    st.markdown("**Quick usage snippet**")
    st.code("""\
import joblib
model = joblib.load("model.joblib")
preds = model.predict(X_new)
""")
    if st.button("Start a new project (reset data)"):
        st.session_state.df = st.session_state.df_original.copy() if st.session_state.df_original is not None else None
        st.session_state.model = None
        st.session_state.trained = False
        st.session_state.step = 0
        st.success("Reset ✓")

# ---------------------------
# Router: show step
# ---------------------------
if st.session_state.step == 0:
    step_business()
elif st.session_state.step == 1:
    step_data()
elif st.session_state.step == 2:
    step_eda()
elif st.session_state.step == 3:
    step_clean()
elif st.session_state.step == 4:
    step_model()
elif st.session_state.step == 5:
    step_evaluate()
elif st.session_state.step == 6:
    step_save()

# Footer
st.markdown("<div class='small-muted' style='margin-top:20px'>Tip: this app generates simple, helpful explanations automatically. For production, add cross-validation, stronger preprocessing, hyperparameter tuning and domain checks.</div>", unsafe_allow_html=True)
