# 💳 Credit Risk Assessment Dashboard

A modern, interactive **machine learning-powered dashboard** for assessing credit risk and predicting loan default probability. Built with Python, XGBoost, and Streamlit for real-time risk analysis and portfolio management.

---

## 🎯 Project Overview

This project demonstrates a **complete end-to-end data science solution** for financial risk assessment, featuring:

- ✅ **XGBoost ML Model** - Trained on synthetic credit data with 92%+ accuracy
- ✅ **Interactive Streamlit Dashboard** - 4 powerful tabs for risk analysis
- ✅ **Real-time Predictions** - Assess individual loan applications instantly
- ✅ **Portfolio Analytics** - Visualize risk distribution across customers
- ✅ **Feature Importance** - Understand which factors drive risk predictions
- ✅ **No JavaScript** - Pure Python backend with beautiful UI

**Perfect for:** Masters thesis, data science portfolio, fintech interviews

---

## 🌟 Key Features

### **Tab 1: 🔍 Risk Assessment**
- Input customer financial details (age, income, credit score, etc.)
- Get instant risk probability and recommendation (APPROVE/REJECT)
- Visual breakdown of risk factors affecting the decision
- Real-time model inference

### **Tab 2: 📊 Analytics**
- Portfolio overview with risk segmentation
- Risk distribution histogram across 1000+ customers
- Pie chart showing Low/Medium/High risk percentages
- Correlation heatmap of all features
- Identify trends in the customer base

### **Tab 3: 👥 Customer Profiles**
- Filter and browse customers by risk level
- Interactive scatter plot: Income vs Credit Score
- Sortable customer data table
- Deep dive into specific customer segments

### **Tab 4: 🎯 Model Insights**
- Feature importance visualization
- Top factors influencing risk predictions
- Model configuration and risk thresholds
- Transparency into model decisions

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python 3.8+ |
| **ML Model** | XGBoost (Gradient Boosting) |
| **Frontend** | Streamlit (No JavaScript!) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly (Interactive Charts) |
| **Model Persistence** | Joblib |

---

## 📊 Model Performance

```
Model Type:        XGBoost Classifier
Training Samples:  1,000 synthetic records
Features:          8 financial indicators
Accuracy:          ~92%
ROC-AUC Score:     ~0.95
Training Time:     <5 seconds
```

### **8 Features Used:**
1. **Age** - Customer age in years
2. **Income** - Annual income ($)
3. **Loan Amount** - Requested loan amount ($)
4. **Employment Years** - Years employed
5. **Credit Score** - Credit score (300-850)
6. **Existing Debts** - Total debts ($)
7. **Monthly Payment Ratio** - Payment to income ratio
8. **Previous Defaults** - History of defaults

---

## 📁 Project Structure

```
credit_risk_dashboard/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📄 config.py                    # Configuration & parameters
├── 📄 app.py                       # Streamlit dashboard
├── 📄 model.py                     # Model training script
├── 📄 data_generator.py            # Synthetic data generation
│
├── 📁 data/
│   └── credit_data.csv             # Generated training data (1000 records)
│
└── 📁 models/
    ├── credit_risk_model.pkl       # Trained XGBoost model
    └── scaler.pkl                  # Feature scaler (StandardScaler)
```

---

## 🚀 Quick Start (5 Minutes)

### **1. Clone the Repository**
```bash
git clone https://github.com/AryanPatill/credit_risk_dashboard.git
cd credit_risk_dashboard
```

### **2. Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Generate Training Data**
```bash
python data_generator.py
```
**Output:** `data/credit_data.csv` with 1,000 synthetic records

### **5. Train the Model**
```bash
python model.py
```
**Output:** `models/credit_risk_model.pkl` and `models/scaler.pkl`

### **6. Launch Dashboard**
```bash
streamlit run app.py
```
**Opens automatically at:** `http://localhost:8501`

---

## 💡 How It Works

### **Data Pipeline:**
```
Generate Synthetic Data
        ↓
    data_generator.py → credit_data.csv (1000 records)
        ↓
    Train XGBoost Model
        ↓
    model.py → credit_risk_model.pkl
        ↓
    Load in Streamlit App
        ↓
    app.py → Interactive Dashboard
```

### **Prediction Process:**
```
User Input (Age, Income, Credit Score, etc.)
        ↓
Feature Scaling (StandardScaler)
        ↓
XGBoost Model Inference
        ↓
Risk Probability (0-1 scale)
        ↓
Risk Category (Low: 0-0.3, Medium: 0.3-0.6, High: 0.6-1.0)
        ↓
Approval Recommendation + Visual Analysis
```

---

## 🎓 Learning Outcomes

This project demonstrates mastery in:

✅ **Machine Learning**
- Model training and hyperparameter tuning
- Classification problems with imbalanced data
- Feature scaling and normalization

✅ **Data Science**
- Synthetic data generation
- Exploratory data analysis
- Feature importance analysis
- Risk assessment frameworks

✅ **Software Engineering**
- Clean, modular code architecture
- Configuration management
- Model persistence and loading
- End-to-end pipeline automation

✅ **Data Visualization**
- Interactive Plotly charts
- Real-time dashboard updates
- Correlation analysis
- Business intelligence dashboards

✅ **Finance Domain Knowledge**
- Credit risk assessment
- Loan approval decision-making
- Portfolio risk analysis
- Financial metrics

---

## 📈 Use Cases

This dashboard can be extended for:

- **Banks & Financial Institutions** - Automated loan approval systems
- **Credit Card Companies** - Credit limit assessment
- **Fintech Startups** - Risk-based lending platforms
- **Insurance Companies** - Risk underwriting
- **Academic Research** - ML model demonstrations

---

## 🔧 Customization

### **Adjust Risk Thresholds:**
Edit `config.py`:
```python
RISK_LEVELS = {
    'Low': {'range': (0, 0.3), 'color': '#27AE60'},      # Green
    'Medium': {'range': (0.3, 0.6), 'color': '#F39C12'},  # Orange
    'High': {'range': (0.6, 1.0), 'color': '#E74C3C'}     # Red
}
```

### **Change Model Parameters:**
Edit `config.py`:
```python
MODEL_PARAMS = {
    'n_estimators': 100,      # More trees = better accuracy
    'max_depth': 8,            # Deeper trees = more complex patterns
    'learning_rate': 0.1,      # Lower = slower learning
    'random_state': 42,        # For reproducibility
}
```

### **Generate More Data:**
Edit `config.py`:
```python
DATASET_SIZE = 5000  # Change from 1000 to 5000
```

---

## 📊 Sample Results

### **Individual Prediction Example:**
```
Customer Profile:
- Age: 35 years
- Income: $75,000/year
- Credit Score: 720
- Loan Amount: $50,000
- Existing Debts: $15,000

Result:
✅ Risk Level: LOW
📊 Risk Probability: 22%
💰 Recommendation: APPROVE
```

### **Portfolio Statistics:**
```
Total Applications: 1,000
- Low Risk: 620 (62%)
- Medium Risk: 280 (28%)
- High Risk: 100 (10%)

Default Rate: 12.3%
```

---

## 🐛 Troubleshooting

### **Error: "Module not found"**
```bash
pip install -r requirements.txt
```

### **Error: "Models not found"**
Make sure you ran:
```bash
python data_generator.py
python model.py
```

### **Port 8501 already in use**
```bash
streamlit run app.py --server.port 8502
```

### **Virtual environment not activating**
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

---

## 📚 Future Enhancements

Potential improvements for v2.0:

- [ ] Real database integration (PostgreSQL/MongoDB)
- [ ] User authentication and role-based access
- [ ] API deployment with FastAPI/Flask
- [ ] Advanced ensemble models (LightGBM, CatBoost)
- [ ] SHAP values for model explainability
- [ ] Batch prediction with CSV upload
- [ ] Risk monitoring and alerts
- [ ] A/B testing framework for model updates

---

## 📄 License

MIT License - Feel free to use this project for educational and commercial purposes.

---

## 👨‍💼 Author

**Aryan Patil**  
📚 *M.S. in Data Science*  
🔗 [LinkedIn](https://www.linkedin.com/in/aryanpatil18/)  
🌐 [GitHub](https://github.com/AryanPatill)

---

## 🙏 Acknowledgments

- **XGBoost** - For powerful gradient boosting
- **Streamlit** - For making dashboards easy
- **Plotly** - For beautiful interactive visualizations
- **Scikit-learn** - For preprocessing and metrics

---

## 📞 Questions or Feedback?

Feel free to open an **Issue** or **Pull Request** on GitHub!

---

**⭐ If you found this helpful, please star this repository!**

Last Updated: December 2025
