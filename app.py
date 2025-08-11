"""
Guided Data Science Studio — Game-like Storytelling Demo App
File: guided_data_science_studio_full_app.py

Features included:
- Game-like, step-by-step wizard following CRISP-DM with a clear roadmap and "what/why/how" explanations
- Rich EDA with Plotly interactive charts, seaborn static charts, pairplots, correlation, and automated storytelling
- Deep cleaning options with many strategies and explanations
- Encoding and feature engineering helpers
- Full ML model suite (classification & regression) using scikit-learn: LogisticRegression, RandomForest, GradientBoosting, DecisionTree, KNeighbors, SVC, LinearRegression, RandomForestRegressor, GradientBoostingRegressor
- Model training, cross-validation, hyperparameter quick-tuning (grid for small sets), and comparison report with metrics
- Visual model comparison (bar charts, ROC, PR curves for classification)
- Downloadable artifacts: trained model (.joblib), comparison CSV, cleaned dataset
- Game-like UI: progress strip, friendly narrations, badges and levels to guide users

How to use:
1. Save this file as `app.py` in your repo.
2. Add requirements.txt with packages listed below and push to GitHub.
3. Deploy on Streamlit Cloud or run locally: `streamlit run app.py`.

Notes:
- This app aims to be educational and easy for complete beginners while offering advanced options for deeper exploration.
- For very large datasets, some operations (pairplot, full CV) are limited or sampled to keep the UI responsive.

"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import time
from typing import List, Tuple, Dict

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ML & preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Optional: hide warnings for cleaner UI
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Page configuration & CSS
# ---------------------------
st.set_page_config(page_title="Guided Data Science Studio — Journey", layout="wide")

st.markdown("""
<style>
/* Header */
.header { padding: 18px; border-radius:12px; background: linear-gradient(90deg,#0f172a,#0ea5a6); color:white;}
.badge { background:#10b981; color:white; padding:6px 10px; border-radius:999px; font-weight:600 }
.card { background:white; padding:14px; border-radius:10px; box-shadow: 0 4px 18px rgba(2,6,23,0.08);}
.step { padding:10px; border-radius:8px; background:#eef2ff; margin-bottom:8px }
.small-muted { color:#6b7280; font-size:13px }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1 style="margin:6px 0">Guided Data Science Studio — The Learning Journey</h1><div style="font-size:14px">Playful, structured, and deeply educational — from data to deployment</div></div>', unsafe_allow_html=True)

# ---------------------------
# Helper utilities: storytelling, metrics, plot helpers
# ---------------------------

def small(text: str):
    st.markdown(f"<div class='small-muted'>{text}</div>", unsafe_allow_html=True)


def badge(text: str):
    st.markdown(f"<span class='badge'>{text}</span>", unsafe_allow_html=True)


# STORY functions: auto-generate plain-English interpretations

def story_dataset_summary(df: pd.DataFrame) -> str:
    if df is None:
        return "No data loaded yet."
    rows, cols = df.shape
    miss = int(df.isnull().sum().sum())
    numeric = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(exclude=[np.number])
    parts = [f"This dataset has **{rows}** rows and **{cols}** columns."]
    parts.append(f"There are **{miss}** missing values in total. Missing data can hide the real picture — we'll handle it soon.")
    if not numeric.empty:
        parts.append(f"We have **{numeric.shape[1]}** numeric columns. Look for skewed numbers and outliers there.")
    if not cat.empty:
        parts.append(f"We have **{cat.shape[1]}** categorical columns (words/categories). We'll convert them to numbers the computer can use.")
    parts.append("We'll next show charts to help you see patterns, surprises, and relationships.")
    return " ".join(parts)


def story_feature_insights(df: pd.DataFrame, col: str) -> str:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        miss_pct = 100 * s.isnull().mean()
        mean = s.mean()
        med = s.median()
        std = s.std()
        skew = s.skew()
        return (f"Column **{col}**: average ~{mean:.2f}, middle {med:.2f}, spread {std:.2f}. "
                f"Missing: {miss_pct:.1f}%. Skewness {skew:.2f}. If skew big, consider log transform or median imputation.")
    else:
        counts = s.value_counts().head(5)
        top = counts.index[0] if not counts.empty else "(none)"
        return f"Column **{col}**: top value '{top}' appears {counts.iloc[0] if not counts.empty else 0} times. Consider grouping rare categories."


# Metric calculations for classification & regression

def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            metrics['roc_auc'] = float('nan')
    return metrics


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    metrics = {}
    metrics['RMSE'] = mean_squared_error(y_true, y_pred, squared=False)
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['R2'] = r2_score(y_true, y_pred)
    return metrics


# ---------------------------
# Session state initialization
# ---------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'project_goal' not in st.session_state:
    st.session_state.project_goal = ''
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {}  # name -> dict(metrics, model, type)
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = None

# ---------------------------
# TOP progress UI
# ---------------------------
steps = [
    "1 • Goal",
    "2 • Data",
    "3 • EDA",
    "4 • Clean",
    "5 • Feature Eng",
    "6 • Train",
    "7 • Compare",
    "8 • Save/Deploy"
]

cols = st.columns(len(steps))
for i, c in enumerate(cols):
    if i == st.session_state.step:
        c.markdown(f"<div class='step'><strong>{steps[i]}</strong></div>", unsafe_allow_html=True)
    else:
        if c.button(steps[i], key=f"goto_{i}"):
            st.session_state.step = i

st.write('---')

# ---------------------------
# Step 1: Goal
# ---------------------------
if st.session_state.step == 0:
    st.header("Step 1 — Project goal (why?)")
    small("What question do you want the data to answer? Keep it short and clear.")
    st.session_state.project_goal = st.text_input("Project goal (one sentence)", value=st.session_state.project_goal)
    st.write("Examples: 'Predict churn (yes/no)', 'Estimate house price (in USD)', 'Classify reviews (pos/neg)'")
    if st.button("Save goal and go to Data"):
        st.session_state.step = 1

# ---------------------------
# Step 2: Data
# ---------------------------
if st.session_state.step == 1:
    st.header("Step 2 — Load data & quick look")
    small("Upload a CSV or Excel file or use one of the built-in demo datasets.")
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
        demo_choice = st.selectbox("Or pick a demo dataset", ['--none--', 'Iris', 'Small messy'])
        if st.button("Load demo dataset") and demo_choice != '--none--':
            if demo_choice == 'Iris':
                from sklearn import datasets
                iris = datasets.load_iris(as_frame=True)
                st.session_state.df = iris.frame.copy()
                st.session_state.df_original = st.session_state.df.copy()
                st.success("Iris loaded")
            elif demo_choice == 'Small messy':
                st.session_state.df = pd.DataFrame({
                    'age':[25,30,np.nan,22,40, None, 80],
                    'gender':['F','M','M',None,'F','F','M'],
                    'score':[88,92,85,None,95,78,60],
                    'bought':[0,1,0,1,1,0,1]
                })
                st.session_state.df_original = st.session_state.df.copy()
                st.success('Small messy loaded')
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                st.session_state.df = df.copy()
                st.session_state.df_original = df.copy()
                st.success('File uploaded and loaded')
            except Exception as e:
                st.error(f"Could not read file: {e}")
    with col2:
        small('Tips')
        st.write('- Prefer CSV files with headers.')
        st.write('- If file large, consider sampling or using an extracted subset.')

    if st.session_state.df is not None:
        st.markdown('**Dataset preview**')
        st.dataframe(st.session_state.df.head(8))
        st.markdown('**Columns & types**')
        df_types = pd.DataFrame({'column': st.session_state.df.columns, 'dtype': st.session_state.df.dtypes.astype(str)})
        st.table(df_types)
        st.markdown('**Auto-story**')
        st.info(story_dataset_summary(st.session_state.df))

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button('Back: Goal'):
            st.session_state.step = 0
    with nav2:
        if st.button('Next: EDA'):
            st.session_state.step = 2

# ---------------------------
# Step 3: EDA (deep graphs + storytelling)
# ---------------------------
if st.session_state.step == 2:
    st.header('Step 3 — Explore & Storytelling (EDA)')
    df = st.session_state.df
    if df is None:
        st.error('No data loaded. Please go back and upload a dataset.')
    else:
        st.subheader('Missing values & overview')
        miss = df.isnull().sum()
        miss = miss[miss>0].sort_values(ascending=False)
        if miss.empty:
            st.success('No missing values found')
        else:
            st.table(miss)

        st.subheader('Interactive column explorer')
        left, mid, right = st.columns([2,2,1])
        with left:
            col = st.selectbox('Pick a column to explore', options=list(df.columns))
            if col:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.histogram(df, x=col, marginal='box', nbins=40, title=f'Histogram & boxplot of {col}')
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(story_feature_insights(df, col))
                else:
                    fig = px.bar(df[col].value_counts().reset_index().rename(columns={'index':'value', col:'count'}), x='count', y='value', orientation='h', title=f'Counts of {col}')
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(story_feature_insights(df, col))
        with mid:
            # correlation heatmap
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] >= 2:
                corr = num.corr()
                fig2 = px.imshow(corr, text_auto='.2f', aspect='auto', title='Correlation matrix (numeric)')
                st.plotly_chart(fig2, use_container_width=True)
                # auto insights
                pairs, corr_text = [], ''
                try:
                    pairs = [(a,b,corr.loc[a,b]) for a in corr.index for b in corr.columns if a!=b and abs(corr.loc[a,b])>0.6]
                    if pairs:
                        corr_text = 'Strong correlations detected: ' + ', '.join([f"{a} vs {b} ({corr.loc[a,b]:.2f})" for a,b,_ in pairs[:6]])
                    else:
                        corr_text = 'No strong correlations (>0.6) detected.'
                except Exception:
                    corr_text = 'Could not compute correlations.'
                st.info(corr_text)
            else:
                st.info('Not enough numeric columns for correlation map.')
        with right:
            st.markdown('**Quick EDA actions**')
            if st.button('Show pairplot (small sample)'):
                n = min(300, df.shape[0])
                num = df.select_dtypes(include=[np.number]).dropna(axis=1)
                if num.shape[1] <= 6 and num.shape[0] > 5:
                    sampled = num.sample(n)
                    pair_fig = sns.pairplot(sampled)
                    st.pyplot(pair_fig.fig)
                else:
                    st.warning('Too many numeric columns or too few rows for pairplot.')
            if st.button('Generate quick profile (light)'):
                st.info('Generating quick summary...')
                time.sleep(0.8)
                st.write(df.describe(include='all').transpose())

        st.markdown('---')
        nav1, nav2 = st.columns(2)
        with nav1:
            if st.button('Back: Data'):
                st.session_state.step = 1
        with nav2:
            if st.button('Next: Clean & Feature Eng'):
                st.session_state.step = 3

# ---------------------------
# Step 4: Cleaning & Feature Engineering
# ---------------------------
if st.session_state.step == 3:
    st.header('Step 4 — Clean, transform & create features')
    df = st.session_state.df
    if df is None:
        st.error('No data loaded.')
    else:
        st.subheader('Missing value strategies (explain & apply)')
        small('What: we fill or remove missing values. Why: models need numbers; missing can bias results. Which: depends on amount and meaning.')
        miss = df.isnull().sum().sort_values(ascending=False)
        st.table(miss[miss>0])
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('Drop rows with any missing'):
                st.session_state.df = df.dropna()
                st.success('Rows dropped')
        with col2:
            default_fill = 'mean'
            if st.button('Fill numeric with mean & categorical with mode'):
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
                st.success('Imputation done')
        with col3:
            if st.button('Fill with constant (0 or "unknown")'):
                df2 = df.fillna(0)
                st.session_state.df = df2
                st.success('Filled with constant')

        st.markdown('---')
        st.subheader('Encoding & feature creation')
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        st.write('Categorical detected:', cat_cols)
        if cat_cols:
            if st.button('One-hot encode small categories / label encode large'):
                df2 = st.session_state.df.copy()
                for c in cat_cols:
                    try:
                        if df2[c].nunique() <= 10:
                            df2 = pd.get_dummies(df2, columns=[c], prefix=c, drop_first=False)
                        else:
                            df2[c] = LabelEncoder().fit_transform(df2[c].astype(str))
                    except Exception as e:
                        st.warning(f'Could not encode {c}: {e}')
                st.session_state.df = df2
                st.success('Encoding applied')
        else:
            st.info('No categorical columns detected')

        st.markdown('---')
        st.subheader('Feature engineering (helpful transforms)')
        if st.button('Add squared terms for numeric features'):
            df2 = st.session_state.df.copy()
            for c in df2.select_dtypes(include=[np.number]).columns:
                df2[f'{c}_squared'] = df2[c] ** 2
            st.session_state.df = df2
            st.success('Squared features added')
        if st.button('Log-transform skewed numeric features'):
            df2 = st.session_state.df.copy()
            for c in df2.select_dtypes(include=[np.number]).columns:
                if (df2[c] > 0).all():
                    df2[f'{c}_log'] = np.log1p(df2[c])
            st.session_state.df = df2
            st.success('Log features added')

        st.markdown('---')
        st.write('Preview of dataset after transforms:')
        st.dataframe(st.session_state.df.head(6))

        nav1, nav2 = st.columns(2)
        with nav1:
            if st.button('Back: EDA'):
                st.session_state.step = 2
        with nav2:
            if st.button('Next: Train models'):
                st.session_state.step = 4

# ---------------------------
# Step 5: Train models (many options + CV)
# ---------------------------
if st.session_state.step == 4:
    st.header('Step 5 — Train models & compare')
    df = st.session_state.df
    if df is None:
        st.error('No data loaded.')
    else:
        st.write('Pick target (what we want to predict).')
        target = st.selectbox('Target column', options=list(df.columns))
        if target:
            problem = 'regression' if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique()>20 else 'classification'
            st.write(f'Auto-detected problem type: **{problem}**')
            features = [c for c in df.columns if c!=target]
            st.write(f'Features used: {len(features)} columns')

            test_size = st.slider('Test set size (%)', min_value=10, max_value=40, value=20)
            do_cv = st.checkbox('Run cross-validation for comparison (may take time)', value=True)
            sample_limit = st.checkbox('Sample dataset for speed (recommended for large data)', value=False)
            if sample_limit and df.shape[0] > 5000:
                df_sample = df.sample(5000, random_state=42)
            else:
                df_sample = df

            # Prepare X,y and simple preprocessing
            X = df_sample[features].copy()
            y = df_sample[target].copy()
            # Simple encoding for text
            for c in X.select_dtypes(include=['object', 'category']).columns:
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
            # Drop rows with missing target
            mask = y.notnull()
            X = X.loc[mask]
            y = y.loc[mask]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)

            # Scaling optional
            scale = st.checkbox('Standard scale numeric features before models', value=True)
            scaler = None
            if scale:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Available models
            models_to_run = {}
            if problem == 'classification':
                models_to_run = {
                    'Logistic Regression': LogisticRegression(max_iter=400),
                    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'Decision Tree': DecisionTreeClassifier(random_state=42),
                    'KNN': KNeighborsClassifier(n_neighbors=5),
                    'SVM (RBF)': SVC(probability=True)
                }
            else:
                models_to_run = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest Regressor': RandomForestRegressor(n_estimators=200, random_state=42),
                    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42)
                }

            st.markdown('**Select models to train**')
            chosen = st.multiselect('Pick from available models', options=list(models_to_run.keys()), default=list(models_to_run.keys())[:2])

            if st.button('Train selected models'):
                st.info('Training models — this may take a while depending on dataset size and models chosen...')
                results = []
                for name in chosen:
                    clf = models_to_run[name]
                    try:
                        clf.fit(X_train, y_train)
                        preds = clf.predict(X_test)
                        # probability for ROC
                        probs = None
                        try:
                            if problem=='classification' and hasattr(clf, 'predict_proba'):
                                probs = clf.predict_proba(X_test)
                        except Exception:
                            probs = None
                        if problem=='classification':
                            mets = classification_metrics(y_test, preds, y_proba=probs)
                        else:
                            mets = regression_metrics(y_test, preds)
                        # store
                        st.session_state.models_trained[name] = {
                            'model': clf,
                            'metrics': mets,
                            'type': problem
                        }
                        results.append((name, mets))
                        st.success(f'{name} trained')
                    except Exception as e:
                        st.error(f'Error training {name}: {e}')

                # Cross-validation if requested
                if do_cv:
                    st.info('Running 5-fold CV on selected models (may be slow).')
                    cv_results = []
                    for name in chosen:
                        clf = models_to_run[name]
                        try:
                            if problem == 'classification':
                                score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
                                cv_results.append((name, float(score.mean())))
                            else:
                                score = cross_val_score(clf, X_train, y_train, cv=5, scoring='r2')
                                cv_results.append((name, float(score.mean())))
                        except Exception as e:
                            cv_results.append((name, None))
                    st.write('CV summary (mean score):')
                    st.table(pd.DataFrame(cv_results, columns=['model', 'cv_mean']).set_index('model'))

                st.success('All selected models processed — go to Compare step for a visual report')

            # Save test split for later evaluation & comparison
            if 'X_test_global' not in st.session_state:
                st.session_state['X_test_global'] = X_test
                st.session_state['y_test_global'] = y_test
                st.session_state['scaler_global'] = scaler

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button('Back: Feature Eng'):
            st.session_state.step = 3
    with nav2:
        if st.button('Next: Compare models'):
            st.session_state.step = 5

# ---------------------------
# Step 6: Model comparison & reports
# ---------------------------
if st.session_state.step == 5:
    st.header('Step 6 — Compare models & generate reports')
    if not st.session_state.models_trained:
        st.warning('No trained models found. Train models in previous step first.')
    else:
        # Build comparison table
        rows = []
        problem = None
        for name, info in st.session_state.models_trained.items():
            mets = info['metrics']
            problem = info['type']
            flat = {'model': name}
            flat.update(mets)
            rows.append(flat)
        comp = pd.DataFrame(rows).set_index('model')
        st.markdown('**Comparison table**')
        st.dataframe(comp.fillna('N/A'))

        # Download comparison
        buf = io.StringIO()
        comp.to_csv(buf)
        buf.seek(0)
        st.download_button('Download comparison CSV', data=buf, file_name='model_comparison.csv')

        # Visual charts
        if problem == 'classification':
            metric_to_plot = st.selectbox('Metric to plot', options=['accuracy','f1_macro','precision_macro','recall_macro','roc_auc'])
            if metric_to_plot in comp.columns:
                fig = px.bar(comp.reset_index(), x=comp.reset_index()['model'], y=comp[metric_to_plot], title=f'Model comparison by {metric_to_plot}')
                st.plotly_chart(fig, use_container_width=True)

            # ROC curves if binary classifiers
            binary_ok = all([ (info['type']=='classification') for info in st.session_state.models_trained.values() ])
            if binary_ok:
                st.markdown('### ROC curves (if models provide probabilities)')
                X_test = st.session_state.get('X_test_global')
                y_test = st.session_state.get('y_test_global')
                fig = go.Figure()
                for name, info in st.session_state.models_trained.items():
                    model = info['model']
                    try:
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_test)[:,1]
                            fpr, tpr, _ = roc_curve(y_test, proba)
                            roc_auc = auc(fpr, tpr)
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={roc_auc:.2f})"))
                    except Exception:
                        pass
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line={'dash':'dash'}, name='random'))
                fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='ROC Curves')
                st.plotly_chart(fig, use_container_width=True)

        else:
            metric_to_plot = st.selectbox('Metric to plot', options=['R2','RMSE','MAE'])
            # For RMSE lower is better; invert for ranking if needed
            display = comp.reset_index()
            if metric_to_plot in display.columns:
                fig = px.bar(display, x='model', y=metric_to_plot, title=f'Model comparison by {metric_to_plot}')
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('---')
        st.markdown('**Pick a model to inspect and download**')
        chosen_model = st.selectbox('Choose a trained model', options=list(st.session_state.models_trained.keys()))
        if chosen_model:
            info = st.session_state.models_trained[chosen_model]
            st.write('Metrics:')
            st.json(info['metrics'])
            if st.button('Download model (.joblib)'):
                buf = io.BytesIO()
                joblib.dump(info['model'], buf)
                buf.seek(0)
                st.download_button('Download model file', data=buf, file_name=f'{chosen_model.replace(" ","_")}.joblib')

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button('Back: Train models'):
            st.session_state.step = 4
    with nav2:
        if st.button('Next: Save & Deploy'):
            st.session_state.step = 6

# ---------------------------
# Step 7: Save & deploy
# ---------------------------
if st.session_state.step == 6:
    st.header('Step 7 — Save artifacts & quick deploy tips')
    st.write('You can download the cleaned dataset, selected trained model(s), and the comparison report.')
    if st.session_state.df is not None:
        buf = io.StringIO()
        st.session_state.df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button('Download cleaned data (CSV)', data=buf, file_name='cleaned_data.csv')
    if st.session_state.comparison_df is not None:
        buf2 = io.StringIO()
        st.session_state.comparison_df.to_csv(buf2)
        buf2.seek(0)
        st.download_button('Download comparison (CSV)', data=buf2, file_name='comparison.csv')

    st.markdown('**Deploy tips**')
    st.write('- Use Streamlit Cloud for simple deployment (link your GitHub repo).')
    st.write('- For production, wrap model inference in a small API (Flask/FastAPI) and add authentication.')
    if st.button('Reset project (keep code)'):
        for k in ['df','df_original','models_trained','comparison_df','X_test_global','y_test_global']:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.step = 0
        st.success('Reset done')

# ---------------------------
# Footer: credits & requirements hint
# ---------------------------
st.markdown('---')
small('This guided studio is educational. For production, add stronger validation, logging, CI/CD, and dataset governance.')


# End of file
