"""
AHADU BANK AI DRIVEN CUSTOMER & PRODUCT GROWTH OPTIMIZATION PLATFORM
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import joblib
import sys
import importlib
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent


def _enable_sklearn_loss_compat():
    """Enable compatibility for old sklearn pickles that reference top-level _loss."""
    try:
        if "_loss" not in sys.modules:
            loss_module = importlib.import_module("sklearn._loss.loss")
            sys.modules["_loss"] = loss_module
            sys.modules["_loss.loss"] = loss_module
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ahadu Bank AI Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
            
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODELS (with caching)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_data_and_models():
    """Load all data and models"""
    try:
        # Adjust paths to match your local environment
        csv_path = BASE_DIR / "ahadu_bank_enriched_real.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            st.warning(f"⚠️ Data file not found at {csv_path}")
            df = pd.DataFrame()
        
        # Try to load models if they exist - check multiple possible paths
        models = {}
        model_paths = {
            "clv": [BASE_DIR / "Model" / "clv_model.pkl", BASE_DIR / "outputs_real" / "clv_model.pkl"],
            "churn": [BASE_DIR / "Model" / "churn_model.pkl", BASE_DIR / "outputs_real" / "churn_model.pkl"],
            "dormancy": [BASE_DIR / "Model" / "dormancy_model.pkl", BASE_DIR / "outputs_real" / "dormancy_model.pkl"],
        }
        
        missing_models = []
        for name, paths in model_paths.items():
            found = False
            for path in paths:
                if Path(path).exists():
                    try:
                        _enable_sklearn_loss_compat()
                        models[name] = joblib.load(path)
                        found = True
                        break
                    except Exception as e:
                        st.warning(f"⚠️ Failed to load model '{name}' from {path}: {e}")
            if not found:
                missing_models.append(name)
        
        return df, models, missing_models
    except Exception as e:
        return pd.DataFrame(), {}, ["clv", "churn", "dormancy"]

df, models, missing_models = load_data_and_models()

# Check if data is loaded
if df.empty:
    st.error("❌ No data loaded. Please ensure the CSV file is in the correct location.")
    st.stop()

# Show warning if models are missing
if missing_models:
    st.warning(
        f"⚠️ **Missing Models**: {', '.join(missing_models)}\n\n"
        f"To generate models, run the Jupyter notebook:\n"
        f"`AI_ML__Driven__C&P_Growth_Optimization_Platform.ipynb`\n\n"
        f"This will create the required model files in the `outputs_real/` folder."
    )

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_regions_and_products(df):
    """Extract regions and products from dataframe"""
    regions = sorted(df["Region"].dropna().unique().tolist()) if "Region" in df.columns else []
    products = sorted(df["Product Type"].dropna().unique().tolist()) if "Product Type" in df.columns else []
    return regions, products

BLUE = "#1f77b4"
GOLD = "#FFC300"
RED = "#D62728"
GREEN = "#2ca02c"

def plot_theme():
    return {
        "template": "plotly_white",
        "font": {"family": "Arial, sans-serif", "size": 12},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        "margin": {"l": 30, "r": 30, "t": 40, "b": 30},
    }

def predict_clv_model(clv_model, age, sex_enc, reg_enc, prod_enc, log_init, log_bal, 
                      tenure_years, days_since_txn, rfm_score, balance_growth_rate, 
                      product_count, dormant, inoperative, active, debit_freez, credit_freez):
    """Predict CLV"""
    try:
        input_data = np.array([[
            age, sex_enc, reg_enc, prod_enc,
            log_init, log_bal, tenure_years,
            days_since_txn, rfm_score, 3, 3, 3,
            balance_growth_rate, (log_bal / (log_init + 1)), product_count,
            int(tenure_years * 365.25), dormant, inoperative, active,
            debit_freez, credit_freez
        ]])
        return clv_model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0

def predict_churn_model(churn_model, age, sex_enc, reg_enc, prod_enc, log_bal, 
                        log_init, tenure_years, days_since_txn, rfm_score, 
                        balance_growth_rate, clv_predicted, debit_freez, 
                        credit_freez, product_count):
    """Predict churn probability"""
    try:
        input_data = np.array([[
            age, sex_enc, reg_enc, prod_enc,
            log_bal, log_init, tenure_years,
            days_since_txn, rfm_score, 3, 3, 3,
            balance_growth_rate, (log_bal / (log_init + 1)), clv_predicted,
            debit_freez, credit_freez, product_count
        ]])
        return churn_model.predict_proba(input_data)[0][1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0

def predict_dormancy_model(dormancy_model, age, sex_enc, reg_enc, prod_enc, 
                           log_bal, tenure_years, days_since_txn, balance_growth_rate, 
                           rfm_score, churn_probability, debit_freez, credit_freez):
    """Predict dormancy risk"""
    try:
        input_data = np.array([[
            age, sex_enc, reg_enc, prod_enc,
            log_bal, tenure_years, days_since_txn,
            balance_growth_rate, rfm_score, churn_probability,
            debit_freez, credit_freez
        ]])
        return dormancy_model.predict_proba(input_data)[0][1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0


# ─────────────────────────────────────────────────────────────
# EDA: Generate and save dashboard image
# ─────────────────────────────────────────────────────────────
@st.cache_data
def create_eda_dashboard(df: pd.DataFrame, out_path: str = "outputs_real/eda_dashboard_real.png") -> str:
    """Create EDA dashboard image and save to out_path. Returns path."""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle("AHADU BANK AI DRIVEN CUSTOMER & PRODUCT GROWTH OPTIMIZATION PLATFORM DASHBOARD ", fontsize=16, fontweight="bold")

    # Age Distribution
    if "Age" in df.columns:
        axes[0,0].hist(df["Age"].dropna(), bins=30, color="#1f77b4", edgecolor="white")
        axes[0,0].set_title("Age Distribution"); axes[0,0].set_xlabel("Age")
    else:
        axes[0,0].text(0.5,0.5,"No Age data", ha='center')

    # Gender Distribution
    if "Sex" in df.columns:
        sx = df["Sex"].value_counts()
        axes[0,1].pie(sx, labels=sx.index, autopct="%1.1f%%", colors=["#4C72B0","#DD8452"])
        axes[0,1].set_title("Gender Distribution")
    else:
        axes[0,1].text(0.5,0.5,"No Sex data", ha='center')

    # Account Status Breakdown
    if "Account_Status_Code" in df.columns:
        status_map = {0:"Active",1:"Inoperative",2:"Dormant",3:"Frozen"}
        sc = df["Account_Status_Code"].map(status_map).value_counts()
        axes[0,2].bar(sc.index, sc.values, color=["#2ca02c","#ff7f0e","#d62728","#9467bd"])
        axes[0,2].set_title("Account Status Breakdown")
    else:
        axes[0,2].text(0.5,0.5,"No Account Status", ha='center')

    # Top 10 Products
    if "Product Type" in df.columns:
        tp = df["Product Type"].value_counts().head(10)
        axes[0,3].barh(tp.index[::-1], tp.values[::-1], color="#17becf")
        axes[0,3].set_title("Top 10 Products")
    else:
        axes[0,3].text(0.5,0.5,"No Product data", ha='center')

    # Log(End Balance)
    if "End Balance" in df.columns:
        axes[1,0].hist(np.log1p(df["End Balance"].fillna(0)), bins=40, color="#e377c2", edgecolor="white")
        axes[1,0].set_title("Log(End Balance)")
    else:
        axes[1,0].text(0.5,0.5,"No End Balance", ha='center')

    # Account Tenure
    if "Account_Tenure_Years" in df.columns:
        axes[1,1].hist(df["Account_Tenure_Years"].fillna(0), bins=30, color="#bcbd22", edgecolor="white")
        axes[1,1].set_title("Account Tenure (Years)")
    else:
        axes[1,1].text(0.5,0.5,"No Tenure data", ha='center')

    # Days Since Last Txn
    if "Days_Since_Last_Txn" in df.columns:
        axes[1,2].hist(df["Days_Since_Last_Txn"].clip(0,1000).fillna(0), bins=40, color="#8c564b", edgecolor="white")
        axes[1,2].set_title("Days Since Last Txn (capped 1000)")
    else:
        axes[1,2].text(0.5,0.5,"No Days_Since_Last_Txn", ha='center')

    # RFM Segment Distribution
    if "RFM_Segment" in df.columns:
        rfm_c = df["RFM_Segment"].value_counts()
        axes[1,3].bar(rfm_c.index.astype(str), rfm_c.values, color=["#d62728","#ff7f0e","#2ca02c","#1f77b4"])
        axes[1,3].set_title("RFM Segment Distribution")
    else:
        axes[1,3].text(0.5,0.5,"No RFM Segment", ha='center')

    # Customers by Region
    if "Region" in df.columns:
        rc = df["Region"].value_counts()
        axes[2,0].barh(rc.index[::-1], rc.values[::-1], color="#7f7f7f")
        axes[2,0].set_title("Customers by Region")
    else:
        axes[2,0].text(0.5,0.5,"No Region data", ha='center')

    # Median Balance by Age Group
    if "Age_Group" in df.columns and "End Balance" in df.columns:
        ab = df.groupby("Age_Group", observed=True)["End Balance"].median()
        axes[2,1].bar(ab.index.astype(str), ab.values, color="#1f77b4")
        axes[2,1].set_title("Median Balance by Age Group")
    else:
        axes[2,1].text(0.5,0.5,"No Age_Group or End Balance", ha='center')

    # Correlation heatmap
    num_cols = [c for c in ["Age","Initial_Deposit","End Balance","Account_Tenure_Years",
                             "Days_Since_Last_Txn","RFM_Score","Balance_Growth_Rate"] if c in df.columns]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        im = axes[2,2].imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[2,2].set_xticks(range(len(num_cols)))
        axes[2,2].set_yticks(range(len(num_cols)))
        axes[2,2].set_xticklabels([c[:8] for c in num_cols], rotation=45, ha="right")
        axes[2,2].set_yticklabels([c[:8] for c in num_cols])
        axes[2,2].set_title("Correlation Heatmap")
        fig.colorbar(im, ax=axes[2,2])
    else:
        axes[2,2].text(0.5,0.5,"Insufficient numeric cols", ha='center')

    # Log(CLV Target)
    if "CLV_Target" in df.columns:
        axes[2,3].hist(np.log1p(df["CLV_Target"].fillna(0)), bins=40, color="#aec7e8", edgecolor="white")
        axes[2,3].set_title("Log(CLV Target)")
    else:
        axes[2,3].text(0.5,0.5,"No CLV Target", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return str(Path(out_path).resolve())

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
# st.sidebar.title("🏦  AI Driven Customer and Product Growth Optimization Platform")

with st.sidebar:
    st.sidebar.image(r"D:\ML-Customer-Growth-Platform\ML-Customer-Growth-Platform\imag\logo.png", width=170)
    st.markdown("""
    <h2 style='text-align:center; color:black;'>
    AI Driven Customer and Product Growth Optimization Platform
    </h2>
    <hr>
    """, unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Module",
    [
        "💰 CLV Prediction",
        "⚠️ Churn Risk",
        "😴 Dormancy Risk",
        "🎯 NBP Recommender",
        "🔄 Re-engagement",
        "📊 Dashboard"
    ]
)


st.sidebar.markdown("---")
st.sidebar.info(
    "🔬 **AI-Driven Customer & Product Growth Optimization**\n\n"
    "This platform uses machine learning to:\n"
    " Predict customer lifetime value (CLV)\n"
    " Identify churn risk\n"
    "Detect dormancy patterns\n"
    " Recommend next best products\n"
    " Drive re-engagement campaigns"
)

regions, products = get_regions_and_products(df)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: CLV PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
if page == "💰 CLV Prediction":
    st.title("💰 Customer Lifetime Value (CLV) Predictor")
    st.markdown("Predict the lifetime value of a customer based on their profile and behavior.")
    
    if "clv" not in models:
        st.error("❌ CLV model not loaded")
        st.info(
            """
            ### 📋 Setup Instructions:
            
            1. **Run the Jupyter Notebook** to train and generate models:
               - Open: `AI_ML__Driven__C&P_Growth_Optimization_Platform.ipynb`
               - Run all cells through to Cell 14
            
            2. **Models will be saved** to `outputs_real/` folder:
               - `clv_model.pkl`
               - `churn_model.pkl`
               - `dormancy_model.pkl`
            
            3. **Restart Streamlit** to load the new models:
               - Press `Ctrl+C` to stop
               - Run: `streamlit cache clear`
               - Run: `streamlit run streamlit_app.py`
            """
        )
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            region = st.selectbox("Region", ["Select..."] + regions, key="clv_region")
        
        with col2:
            product = st.selectbox("Product Type", ["Select..."] + products, key="clv_product")
            initial_deposit = st.number_input("Initial Deposit (ETB)", min_value=0.0, value=5000.0)
            end_balance = st.number_input("End Balance (ETB)", min_value=0.0, value=120000.0)
        
        with col3:
            tenure_years = st.number_input("Account Tenure (Years)", min_value=0.0, max_value=50.0, value=2.5)
            days_since_txn = st.number_input("Days Since Last Txn", min_value=0, value=45)
            rfm_score = st.number_input("RFM Score (3-15)", min_value=3, max_value=15, value=9)
        
        col4, col5 = st.columns(2)
        with col4:
            balance_growth_rate = st.number_input("Balance Growth Rate", min_value=-1.0, max_value=10.0, value=0.4)
            product_count = st.number_input("Product Count", min_value=1, max_value=10, value=1)
        
        with col5:
            dormant = st.radio("Dormant", [0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")
            inoperative = st.radio("Inoperative", [0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")
            active = st.radio("Active", [0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")
        
        col6, col7 = st.columns(2)
        with col6:
            debit_freez = st.radio("Debit Freeze", [0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")
        with col7:
            credit_freez = st.radio("Credit Freeze", [0, 1], horizontal=True, format_func=lambda x: "No" if x == 0 else "Yes")
        
        if st.button("🔮 Predict CLV", key="predict_clv", use_container_width=True):
            if region == "Select..." or product == "Select...":
                st.error("Please select Region and Product Type")
            else:
                try:
                    sex_enc = 1 if sex == "Male" else 0
                    le_reg = LabelEncoder().fit(df["Region"].dropna().unique())
                    le_prod = LabelEncoder().fit(df["Product Type"].dropna().unique())
                    
                    reg_enc = le_reg.transform([region])[0] if region in le_reg.classes_ else 0
                    prod_enc = le_prod.transform([product])[0] if product in le_prod.classes_ else 0
                    
                    log_init = np.log1p(initial_deposit)
                    log_bal = np.log1p(end_balance)
                    
                    clv_val = predict_clv_model(
                        models["clv"], age, sex_enc, reg_enc, prod_enc, log_init, log_bal,
                        tenure_years, days_since_txn, rfm_score, balance_growth_rate,
                        product_count, dormant, inoperative, active, debit_freez, credit_freez
                    )
                    
                    # Determine tier
                    quantiles = df["CLV_Predicted"].quantile([0.25, 0.5, 0.75])
                    if clv_val >= quantiles[0.75]:
                        tier = "🥇 Top-CLV"
                        color = "green"
                    elif clv_val >= quantiles[0.5]:
                        tier = "🥈 High-CLV"
                        color = "blue"
                    elif clv_val >= quantiles[0.25]:
                        tier = "🥉 Mid-CLV"
                        color = "orange"
                    else:
                        tier = "⬇️ Low-CLV"
                        color = "red"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("💰 Predicted CLV", f"ETB {clv_val:,.2f}")
                    with col2:
                        st.metric("🏷️ CLV Tier", tier)
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>✅ Prediction Complete</h4>
                    This customer has a predicted CLV of <strong>ETB {clv_val:,.2f}</strong> and falls 
                    into the <strong>{tier}</strong> category based on their profile.
                    </div>
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: CHURN RISK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Churn Risk":
    st.title("⚠️ Churn Risk Predictor")
    st.markdown("Identify customers at risk of becoming dormant.")
    
    if "churn" not in models:
        st.error("❌ Churn model not loaded")
        st.info(
            """
            ### 📋 Setup Instructions:
            
            1. **Run the Jupyter Notebook** to train and generate models:
               - Open: `AI_ML__Driven__C&P_Growth_Optimization_Platform.ipynb`
               - Run all cells through to Cell 14
            
            2. **Models will be saved** to `outputs_real/` folder:
               - `clv_model.pkl`
               - `churn_model.pkl`
               - `dormancy_model.pkl`
            
            3. **Restart Streamlit** to load the new models:
               - Press `Ctrl+C` to stop
               - Run: `streamlit cache clear`
               - Run: `streamlit run streamlit_app.py`
            """
        )
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, key="churn_age")
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True, key="churn_sex")
            region = st.selectbox("Region", ["Select..."] + regions, key="churn_region")
        
        with col2:
            product = st.selectbox("Product Type", ["Select..."] + products, key="churn_product")
            initial_deposit = st.number_input("Initial Deposit (ETB)", min_value=0.0, value=5000.0, key="churn_init")
            end_balance = st.number_input("End Balance (ETB)", min_value=0.0, value=12000.0, key="churn_bal")
        
        with col3:
            tenure_years = st.number_input("Account Tenure (Years)", min_value=0.0, max_value=50.0, value=2.5, key="churn_tenure")
            days_since_txn = st.number_input("Days Since Last Txn", min_value=0, value=45, key="churn_days")
            rfm_score = st.number_input("RFM Score (3-15)", min_value=3, max_value=15, value=9, key="churn_rfm")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            balance_growth_rate = st.number_input("Balance Growth Rate", min_value=-1.0, max_value=10.0, value=0.4, key="churn_bgr")
            clv_predicted = st.number_input("CLV Predicted (ETB)", min_value=0.0, value=50000.0, key="churn_clv")
        
        with col5:
            debit_freez = st.radio("Debit Freeze", [0, 1], horizontal=True, key="churn_debit")
            credit_freez = st.radio("Credit Freeze", [0, 1], horizontal=True, key="churn_credit")
        
        with col6:
            product_count = st.number_input("Product Count", min_value=1, max_value=10, value=1, key="churn_pcount")
        
        if st.button("🔮 Predict Churn Risk", key="predict_churn", use_container_width=True):
            if region == "Select..." or product == "Select...":
                st.error("Please select Region and Product Type")
            else:
                try:
                    sex_enc = 1 if sex == "Male" else 0
                    le_reg = LabelEncoder().fit(df["Region"].dropna().unique())
                    le_prod = LabelEncoder().fit(df["Product Type"].dropna().unique())
                    
                    reg_enc = le_reg.transform([region])[0] if region in le_reg.classes_ else 0
                    prod_enc = le_prod.transform([product])[0] if product in le_prod.classes_ else 0
                    
                    log_init = np.log1p(initial_deposit)
                    log_bal = np.log1p(end_balance)
                    
                    churn_prob = predict_churn_model(
                        models["churn"], age, sex_enc, reg_enc, prod_enc, log_bal, log_init,
                        tenure_years, days_since_txn, rfm_score, balance_growth_rate,
                        clv_predicted, debit_freez, credit_freez, product_count
                    )
                    
                    # Determine risk band
                    if churn_prob > 0.70:
                        risk_band = "🔴 Critical Risk"
                        color = "red"
                    elif churn_prob > 0.50:
                        risk_band = "🟠 High Risk"
                        color = "orange"
                    elif churn_prob > 0.30:
                        risk_band = "🟡 Medium Risk"
                        color = "yellow"
                    else:
                        risk_band = "🟢 Low Risk"
                        color = "green"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("⚠️ Churn Probability", f"{churn_prob*100:.1f}%")
                    with col2:
                        st.metric("🚦 Risk Band", risk_band)
                    
                    # Visualization
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=churn_prob*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Churn Risk %"},
                        delta={'reference': 50},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': color},
                               'steps': [
                                   {'range': [0, 30], 'color': "#f7f0ef"},
                                   {'range': [30, 50], 'color': "#fff3cd"},
                                   {'range': [50, 70], 'color': "#ffe5e5"},
                                   {'range': [70, 100], 'color': "#f8d7da"}
                               ]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: DORMANCY RISK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "😴 Dormancy Risk":
    st.title("😴 Dormancy Risk Detector")
    st.markdown("Detect accounts at risk of becoming inoperative.")
    
    if "dormancy" not in models:
        st.error("❌ Dormancy model not loaded")
        st.info(
            """
            ### 📋 Setup Instructions:
            
            1. **Run the Jupyter Notebook** to train and generate models:
               - Open: `AI_ML__Driven__C&P_Growth_Optimization_Platform.ipynb`
               - Run all cells through to Cell 14
            
            2. **Models will be saved** to `outputs_real/` folder:
               - `clv_model.pkl`
               - `churn_model.pkl`
               - `dormancy_model.pkl`
            
            3. **Restart Streamlit** to load the new models:
               - Press `Ctrl+C` to stop
               - Run: `streamlit cache clear`
               - Run: `streamlit run streamlit_app.py`
            """
        )
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, key="dorm_age")
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True, key="dorm_sex")
            region = st.selectbox("Region", ["Select..."] + regions, key="dorm_region")
        
        with col2:
            product = st.selectbox("Product Type", ["Select..."] + products, key="dorm_product")
            end_balance = st.number_input("End Balance (ETB)", min_value=0.0, value=12000.0, key="dorm_bal")
            tenure_years = st.number_input("Account Tenure (Years)", min_value=0.0, max_value=50.0, value=2.5, key="dorm_tenure")
        
        with col3:
            days_since_txn = st.number_input("Days Since Last Txn", min_value=0, value=45, key="dorm_days")
            rfm_score = st.number_input("RFM Score (3-15)", min_value=3, max_value=15, value=9, key="dorm_rfm")
            balance_growth_rate = st.number_input("Balance Growth Rate", min_value=-1.0, max_value=10.0, value=0.4, key="dorm_bgr")
        
        col4, col5 = st.columns(2)
        with col4:
            churn_probability = st.slider("Churn Probability (0-1)", 0.0, 1.0, 0.2, key="dorm_churn")
            debit_freez = st.radio("Debit Freeze", [0, 1], horizontal=True, key="dorm_debit")
        with col5:
            credit_freez = st.radio("Credit Freeze", [0, 1], horizontal=True, key="dorm_credit")
        
        if st.button("🔮 Predict Dormancy Risk", key="predict_dormancy", use_container_width=True):
            if region == "Select..." or product == "Select...":
                st.error("Please select Region and Product Type")
            else:
                try:
                    sex_enc = 1 if sex == "Male" else 0
                    le_reg = LabelEncoder().fit(df["Region"].dropna().unique())
                    le_prod = LabelEncoder().fit(df["Product Type"].dropna().unique())
                    
                    reg_enc = le_reg.transform([region])[0] if region in le_reg.classes_ else 0
                    prod_enc = le_prod.transform([product])[0] if product in le_prod.classes_ else 0
                    log_bal = np.log1p(end_balance)
                    
                    dorm_score = predict_dormancy_model(
                        models["dormancy"], age, sex_enc, reg_enc, prod_enc, log_bal,
                        tenure_years, days_since_txn, balance_growth_rate, rfm_score,
                        churn_probability, debit_freez, credit_freez
                    )
                    
                    # Determine risk band
                    if dorm_score > 0.60:
                        risk_band = "🔴 High Dormancy Risk"
                        color = "red"
                    elif dorm_score > 0.30:
                        risk_band = "🟡 At Risk"
                        color = "orange"
                    else:
                        risk_band = "🟢 Safe"
                        color = "green"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("😴 Dormancy Risk Score", f"{dorm_score*100:.1f}%")
                    with col2:
                        st.metric("🏷️ Risk Band", risk_band)
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>✅ Assessment Complete</h4>
                    This account has a dormancy risk score of <strong>{dorm_score*100:.1f}%</strong> 
                    and is classified as <strong>{risk_band}</strong>.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: NBP RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎯 NBP Recommender":
    st.title("🎯 Next Best Product (NBP) Recommender")
    st.markdown("Get personalized product recommendations based on customer profile and behavior.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=35, key="nbp_age")
        region = st.selectbox("Region", ["Select..."] + regions, key="nbp_region")
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True, key="nbp_sex")
    
    with col2:
        product = st.selectbox("Current Product", ["Select..."] + products, key="nbp_product")
        end_balance = st.number_input("End Balance (ETB)", min_value=0.0, value=12000.0, key="nbp_bal")
        tenure_years = st.number_input("Account Tenure (Years)", min_value=0.0, max_value=50.0, value=2.5, key="nbp_tenure")
    
    with col3:
        days_since_txn = st.number_input("Days Since Last Txn", min_value=0, value=45, key="nbp_days")
        rfm_score = st.number_input("RFM Score (3-15)", min_value=3, max_value=15, value=9, key="nbp_rfm")
        clv_predicted = st.number_input("CLV Predicted (ETB)", min_value=0.0, value=5000.0, key="nbp_clv")
    
    churn_probability = st.slider("Churn Probability (0-1)", 0.0, 1.0, 0.2, key="nbp_churn")
    
    if st.button("🎯 Get Recommendations", key="recommend_nbp", use_container_width=True):
        if region == "Select..." or product == "Select...":
            st.error("Please select Region and Current Product")
        else:
            try:
                # Determine segments
                rfm_segment = "Champions" if rfm_score >= 12 else \
                              "High Value" if rfm_score >= 9 else \
                              "Mid Value" if rfm_score >= 6 else "Low Value"
                
                quantiles = df["CLV_Predicted"].quantile([0.25, 0.5, 0.75])
                clv_tier = "Top-CLV" if clv_predicted >= quantiles[0.75] else \
                           "High-CLV" if clv_predicted >= quantiles[0.5] else \
                           "Mid-CLV" if clv_predicted >= quantiles[0.25] else "Low-CLV"
                
                age_group = "18-25" if age <= 25 else \
                            "26-35" if age <= 35 else \
                            "36-45" if age <= 45 else \
                            "46-55" if age <= 55 else "56+"
                
                churn_band = "Critical Risk" if churn_probability > 0.70 else \
                             "High Risk" if churn_probability > 0.50 else \
                             "Medium Risk" if churn_probability > 0.30 else "Low Risk"
                
                # Display profile
                st.markdown("### 👤 Customer Profile")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Age Group", age_group)
                col2.metric("RFM Segment", rfm_segment)
                col3.metric("CLV Tier", clv_tier)
                col4.metric("Churn Risk", churn_band)
                
                # Simple product recommendation logic
                product_list = products if products else ["Saving Account", "Loan", "Investment"]
                available_products = [p for p in product_list if p != product][:3]
                
                # st.markdown("### 🎯 Top Recommendations")
                # for i, rec_product in enumerate(available_products[:3], 1):
                #     medal = ["🥇", "🥈", "🥉"][i-1]
                #     confidence = (1 - i*0.15) * 100
                #     st.success(f"{medal} **Rank {i}: {rec_product}**  \nConfidence: {confidence:.1f}%")
                
                #++++++++++++++++++++++++++
                st.markdown("### 🎯 Top Recommendations")

                customer_age = age

                recommended_products = []
                if customer_age < 18:
                    recommended_products = [
                        "Blatena Saving Account",
                        "Dekemezmur Saving Account",
                        "Special Saving Account"
                    ]
                elif 18 <= customer_age <= 30:
                    recommended_products = [
                        "Werazut Saving Account",
                        "Prime Customer Deposit Account",
                        "Special Saving Account"
                    ]
                elif customer_age > 50:
                    recommended_products = [
                        "Abew Saving Account",
                        "Prime Customer Deposit Account",
                        "Special Saving Account"
                    ]
                else:
                    recommended_products = [
                        "Special Saving Account",
                        "Prime Customer Deposit Account",
                        "Ahadu Bereket Saving Account"
                    ]

                for i, rec_product in enumerate(recommended_products[:3], start=1):
                    medal = ["🥇", "🥈", "🥉"][i - 1]
                    confidence = (1 - (i - 1) * 0.15) * 100

                    st.success(
                        f"{medal} **Rank {i}: {rec_product}**\n"
                        f"Confidence: {confidence:.1f}%"
                    )

                # Action insight
                st.markdown("### 💡 Action Insight")
                if churn_probability > 0.50:
                    st.markdown(
                        f"<div class='danger-box'>"
                        f"<strong>⚠️ High churn risk!</strong> Prioritize retention offer with "
                        f"{available_products[0] if available_products else 'a premium product'}."
                        f"</div>",
                        unsafe_allow_html=True
                    )
                elif clv_tier in ["Top-CLV", "High-CLV"]:
                    st.markdown(
                        f"<div class='success-box'>"
                        f"<strong>🌟 High-value customer</strong> — offer premium product upgrade: "
                        f"{available_products[0] if available_products else 'premium service'}."
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='success-box'>"
                        f"<strong>📈 Cross-sell opportunity:</strong> {available_products[0] if available_products else 'new product'}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error during recommendations: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: RE-ENGAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔄 Re-engagement":
    st.title("🔄 Dormant Account Re-engagement")
    st.markdown("Identify and prioritize dormant accounts for re-engagement campaigns.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        region_filter = st.selectbox("Filter by Region", ["All Regions"] + regions, key="reeng_region")
        product_filter = st.selectbox("Filter by Product", ["All Products"] + products, key="reeng_product")
    
    with col2:
        min_balance = st.number_input("Minimum Balance (ETB)", min_value=0, value=0, key="reeng_minbal")
        max_days = st.slider("Max Days Inactive", 30, 3650, 1825, step=30, key="reeng_days")
    
    with col3:
        top_n = st.slider("Number to Show", 5, 100, 20, step=5, key="reeng_topn")
        sort_by = st.radio(
            "Sort By",
            ["CLV (Highest)", "Balance (Highest)", "Days Inactive (Most)"],
            horizontal=True,
            key="reeng_sort"
        )
    
    if st.button("🔍 Find Dormant Accounts", key="find_dormant", use_container_width=True):
        try:
            # Filter dormant accounts
            dormant_df = df[df["DORMANT"] == 1].copy() if "DORMANT" in df.columns else df.copy()
            
            if region_filter != "All Regions" and "Region" in dormant_df.columns:
                dormant_df = dormant_df[dormant_df["Region"] == region_filter]
            
            if product_filter != "All Products" and "Product Type" in dormant_df.columns:
                dormant_df = dormant_df[dormant_df["Product Type"] == product_filter]
            
            if "End Balance" in dormant_df.columns:
                dormant_df = dormant_df[dormant_df["End Balance"] >= min_balance]
            
            if "Days_Since_Last_Txn" in dormant_df.columns:
                dormant_df = dormant_df[dormant_df["Days_Since_Last_Txn"] <= max_days]
            
            # Sort
            sort_col = "CLV_Predicted" if "CLV (Highest)" in sort_by else \
                       "End Balance" if "Balance" in sort_by else \
                       "Days_Since_Last_Txn"
            
            if sort_col in dormant_df.columns:
                dormant_df = dormant_df.sort_values(sort_col, ascending=False)
            
            top_df = dormant_df.head(int(top_n))
            
            # Summary metrics
            st.markdown("### 📊 Re-engagement Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            total_found = len(dormant_df)
            total_balance = dormant_df["End Balance"].sum() if "End Balance" in dormant_df.columns else 0
            avg_clv = dormant_df["CLV_Predicted"].mean() if "CLV_Predicted" in dormant_df.columns else 0
            avg_inactive = dormant_df["Days_Since_Last_Txn"].mean() if "Days_Since_Last_Txn" in dormant_df.columns else 0
            
            col1.metric("Total Dormant", f"{total_found:,}")
            col2.metric("Balance at Risk", f"ETB {total_balance:,.0f}")
            col3.metric("Avg CLV", f"ETB {avg_clv:,.0f}")
            col4.metric("Avg Days Inactive", f"{avg_inactive:,.0f}")
            
            # Display table
            st.markdown("### 🏦 Top Dormant Accounts")
            display_cols = ["Account Name", "Region", "Product Type", "End Balance", 
                           "Days_Since_Last_Txn", "CLV_Predicted"]
            display_cols = [c for c in display_cols if c in top_df.columns]
            
            if display_cols:
                display_df = top_df[display_cols].copy()
                display_df.columns = ["Name", "Region", "Product", "Balance (ETB)", 
                                     "Days Inactive", "CLV (ETB)"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Action recommendation
            st.markdown("### 📋 Action Plan")
            if total_found == 0:
                st.info("✅ No dormant accounts match your filters.")
            elif avg_clv > df["CLV_Predicted"].quantile(0.75):
                st.markdown(
                    f"""
                    <div class='danger-box'>
                    <h4>🌟 HIGH-VALUE SEGMENT</h4>
                    <ul>
                    <li>Assign relationship manager</li>
                    <li>Personal call within 48 hours</li>
                    <li>Offer premium product upgrade</li>
                    <li><strong>💰 Recoverable: ETB {total_balance:,.0f}</strong></li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='warning-box'>
                    <h4>📈 STANDARD SEGMENT</h4>
                    <ul>
                    <li>Personalized SMS/email campaign</li>
                    <li>Offer fee waiver or bonus interest</li>
                    <li><strong>💰 Recoverable: ETB {total_balance:,.0f}</strong></li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        except Exception as e:
            st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Business Intelligence Dashboard")
    
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("📊 Total Customers", f"{len(df):,}")
        col2.metric("💰 Avg End Balance", f"ETB {df['End Balance'].mean():,.0f}" if "End Balance" in df.columns else "N/A")
        col3.metric("📈 Active Accounts", f"{df['ACTIVE_ACCOUNT'].sum():,}" if "ACTIVE_ACCOUNT" in df.columns else "N/A")
        col4.metric("⏸️ Dormant Accounts", f"{df['DORMANT'].sum():,}" if "DORMANT" in df.columns else "N/A")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if "CLV_Tier" in df.columns:
                st.subheader("📊 CLV Tier Distribution")
                clv_counts = df["CLV_Tier"].value_counts()
                st.bar_chart(clv_counts)
        
        with col2:
            if "Region" in df.columns:
                st.subheader("🌍 Customers by Region")
                region_counts = df["Region"].value_counts()
                st.bar_chart(region_counts)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "Segment_Label" in df.columns:
                st.subheader("👥 Customer Segments")
                seg_counts = df["Segment_Label"].value_counts()
                st.bar_chart(seg_counts)
        
        with col2:
            if "Churn_Risk_Band" in df.columns:
                st.subheader("⚠️ Churn Risk Distribution")
                churn_counts = df["Churn_Risk_Band"].value_counts()
                st.bar_chart(churn_counts)
        # EDA Dashboard image (generate if missing)
        eda_path = "outputs_real/eda_dashboard_real.png"
        col_eda_btn, col_eda_img = st.columns([1,3])
        with col_eda_btn:
            if st.button("🔄 Regenerate EDA Dashboard"):
                try:
                    saved = create_eda_dashboard(df, eda_path)
                    st.success(f"EDA image generated: {saved}")
                except Exception as e:
                    st.error(f"Failed to generate EDA image: {e}")

        with col_eda_img:
            from pathlib import Path
            if Path(eda_path).exists():
                st.image(eda_path, use_column_width=True, caption="EDA dashboard")
            else:
                if st.button("📈 Generate EDA Dashboard Now"):
                    try:
                        saved = create_eda_dashboard(df, eda_path)
                        st.image(saved, use_column_width=True, caption="EDA dashboard")
                    except Exception as e:
                        st.error(f"Failed to generate EDA image: {e}")

        # Summary Statistics
        st.markdown("### 📈 Key Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "CLV_Predicted" in df.columns:
                st.metric("Avg CLV Predicted", f"ETB {df['CLV_Predicted'].mean():,.0f}")
        
        with col2:
            if "Churn_Probability" in df.columns:
                high_churn = (df["Churn_Probability"] > 0.7).sum()
                st.metric("High Churn Risk", f"{high_churn:,}")
        
        with col3:
            if "Account_Tenure_Years" in df.columns:
                st.metric("Avg Tenure", f"{df['Account_Tenure_Years'].mean():.1f} years")
    
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

    st.markdown("<div class='section-title'>📊 Exploratory Data Analysis</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Product Type", color="Product Type",
                           title="Account Distribution by Product",
                           color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(**plot_theme(), showlegend=False,
                          xaxis_tickangle=-40, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.pie(df, names="Sex", title="Gender Distribution",
                     color_discrete_map={"M": BLUE, "F": GOLD}, hole=0.45)
        fig.update_layout(**plot_theme(), height=380)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(df, x="Age", nbins=30, color_discrete_sequence=[GOLD],
                           title="Age Distribution")
        fig.update_layout(**plot_theme(), height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.box(df, x="ACTIVE_ACCOUNT", y="End Balance",
                     color="ACTIVE_ACCOUNT",
                     title="Balance by Account Status",
                     color_discrete_map={0: RED, 1: GREEN},
                     labels={"ACTIVE_ACCOUNT": "Active (1) / Inactive (0)"},
                     log_y=True)
        fig.update_layout(**plot_theme(), height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Status Breakdown</div>", unsafe_allow_html=True)
    status_df = pd.DataFrame({
        "Status":  ["Active", "Inactive", "Dormant", "Inoperative",
                    "Debit Freeze", "Credit Freeze"],
        "Count":   [df["ACTIVE_ACCOUNT"].sum(),
                    len(df) - df["ACTIVE_ACCOUNT"].sum(),
                    df["DORMANT"].sum(), df["INOPERATIVE"].sum(),
                    df["Debit Freez"].sum(), df["Credit Freez"].sum()],
    })
    fig = px.bar(status_df, x="Status", y="Count",
                 color="Count", color_continuous_scale="Oranges",
                 title="Account Status Overview")
    fig.update_layout(**plot_theme(), height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df.head(200), use_container_width=True, height=350)


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p style='color: #28a745; font-size: 12px;'>
    🏦 Ahadu Bank AI Driven Customer & Product Growth Optimization Platform<br>
    Powered by Machine Learning | © 2026 Ahadu Bank. All rights reserved.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)
