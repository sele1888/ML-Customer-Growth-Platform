# 🏦 Ahadu Bank AI Customer Growth Platform - Streamlit Version

This is a Streamlit web application for the Ahadu Bank AI-Driven Customer & Product Growth Optimization Platform.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Your data file: `ahadu_bank_enriched_real.csv`
- Trained models (generated from the main Jupyter notebook):
  - `outputs_real/clv_model.pkl`
  - `outputs_real/churn_model.pkl`
  - `outputs_real/dormancy_model.pkl`

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly matplotlib seaborn xgboost lightgbm shap
```

### Step 2: Run the Application

Navigate to your project directory and run:

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
ML-Customer-Growth-Platform-Development/
├── AI_ML__Driven__C&P_Growth_Optimization_Platform.ipynb  (Main ML pipeline)
├── streamlit_app.py                                        (Streamlit web app)
├── requirements.txt                                        (Python dependencies)
├── ahadu_bank_enriched_real.csv                          (Input data)
├── README_Streamlit.md                                     (This file)
└── outputs_real/                                          (Generated outputs)
    ├── clv_model.pkl                                      (CLV model)
    ├── churn_model.pkl                                    (Churn model)
    ├── dormancy_model.pkl                                 (Dormancy model)
    ├── ahadu_bank_enriched_real.csv                       (Enriched dataset)
    ├── nbp_recommendations_real.csv                       (Recommendations)
    ├── dormant_reengagement_real.csv                      (Reengagement list)
    └── [various charts and outputs]
```

## 🎯 Features

### 1. 💰 CLV Prediction
- Predict customer lifetime value based on customer profile
- Classify customers into CLV tiers (Top-CLV, High-CLV, Mid-CLV, Low-CLV)
- Input customer demographics, account info, and transaction history

### 2. ⚠️ Churn Risk Prediction
- Identify customers at risk of becoming dormant
- Real-time churn probability calculation
- Risk band classification (Critical, High, Medium, Low)
- Visual gauge chart for risk assessment

### 3. 😴 Dormancy Risk Detection
- Detect accounts at risk of inactivity
- Predict dormancy probability
- Identify high-risk accounts for intervention

### 4. 🎯 NBP (Next Best Product) Recommender
- Personalized product recommendations
- Customer profile analysis
- Action insights and recommendations
- Three-tier recommendation system

### 5. 🔄 Dormant Account Re-engagement
- Filter dormant accounts by region, product, balance
- Sort by CLV, balance, or inactivity days
- Identify high-value opportunities
- Generate action plans for re-engagement

### 6. 📊 Business Intelligence Dashboard
- Key metrics and KPIs
- Distribution charts
- Segment analysis
- Regional breakdown

## 🔧 Configuration

### Data File Path
If your CSV file is in a different location, update the `load_data_and_models()` function:

```python
csv_path = "path/to/your/ahadu_bank_enriched_real.csv"
```

### Model File Paths
If your models are in different directories:

```python
model_files = {
    "clv": "path/to/clv_model.pkl",
    "churn": "path/to/churn_model.pkl",
    "dormancy": "path/to/dormancy_model.pkl",
}
```

## 🎨 Customization

### Change Color Scheme
Edit the CSS in the `<style>` section:

```python
st.markdown("""
    <style>
    .success-box {
        background-color: #d4edda;  /* Change color here */
        ...
    }
    </style>
""", unsafe_allow_html=True)
```

### Modify Sidebar Information
Update the sidebar info text:

```python
st.sidebar.info(
    "Your custom info text here..."
)
```

## 📊 Generating Required Models

To generate the required model files, run the main Jupyter notebook:

1. Open `AI_ML__Driven__C&P_Growth_Optimization_Platform.ipynb`
2. Run all cells from Cell 1 to Cell 14
3. Models will be saved to `outputs_real/` directory:
   - `clv_model.pkl`
   - `churn_model.pkl`
   - `dormancy_model.pkl`

## 🐛 Troubleshooting

### Error: "No such file or directory"
- Ensure the CSV file path is correct
- Check that the `outputs_real/` folder exists with model files

### Error: "Model not loaded"
- Run the main Jupyter notebook to generate models
- Verify model files exist in the correct location
- Check file permissions

### Error: "ModuleNotFoundError"
- Install missing packages: `pip install -r requirements.txt`
- Ensure you're using the correct Python environment

### Streamlit won't start
- Try: `streamlit run streamlit_app.py --logger.level=debug`
- Check Python version: `python --version` (should be 3.8+)

## 📈 Performance Tips

- **Caching**: The app uses `@st.cache_resource` for data/model loading
- **Large datasets**: Filter data before displaying
- **Predictions**: Models are loaded once and cached for better performance

## 🔐 Security Notes

- Store sensitive data (API keys, credentials) in environment variables
- Use `.gitignore` to exclude:
  ```
  outputs_real/
  *.pkl
  *.csv
  .streamlit/secrets.toml
  ```

## 📱 Deployment Options

### Local Network
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Streamlit Cloud
1. Push your code to GitHub
2. Deploy at https://share.streamlit.io/
3. Connect your repository

### Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Streamlit documentation: https://docs.streamlit.io/
3. Check model training output from the Jupyter notebook

## 📄 License

Ahadu Bank AI Platform © 2025

---

**Last Updated**: 2025-12-31
**Version**: 1.0
