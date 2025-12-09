# app.py
"""
Credit Risk Dashboard using Streamlit (No JavaScript needed!)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from config import FEATURE_NAMES, RISK_LEVELS, MODEL_PARAMS
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #ff7f0e;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    """Load trained model and scaler"""
    model = joblib.load('models/credit_risk_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

# Load data for analytics
@st.cache_data
def load_data():
    """Load credit dataset"""
    return pd.read_csv('data/credit_data.csv')

model, scaler = load_model()
data = load_data()

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("💳 Credit Risk Assessment Dashboard")
st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Risk Assessment", "📊 Analytics", "👥 Customer Profiles", "🎯 Model Insights"]
)

# ============================================================================
# TAB 1: RISK ASSESSMENT (Interactive Prediction)
# ============================================================================

with tab1:
    st.header("Assess Credit Risk for Individual Application")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=50000)
    
    with col2:
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=1000000, value=50000)
        employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=5)
    
    with col3:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        existing_debts = st.number_input("Existing Debts ($)", min_value=0, max_value=500000, value=10000)
    
    with col4:
        monthly_payment_ratio = st.slider("Monthly Payment Ratio", 0.0, 1.0, 0.3)
        previous_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0)
    
    # Make prediction
    if st.button("🔮 Predict Risk", use_container_width=True):
        # Prepare input
        input_data = np.array([[
            age, income, loan_amount, employment_years,
            credit_score, existing_debts, monthly_payment_ratio, previous_defaults
        ]])
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        risk_probability = model.predict_proba(input_scaled)[0][1]
        prediction = model.predict(input_scaled)[0]
        
        # Determine risk level
        risk_level = "Low"
        for level, params in RISK_LEVELS.items():
            if params['range'][0] <= risk_probability < params['range'][1]:
                risk_level = level
                break
        
        # Display results
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color = RISK_LEVELS[risk_level]['color']
            st.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                <h2 style='margin: 0; color: white;'>{risk_level.upper()}</h2>
                <h3 style='margin: 0; color: white;'>RISK</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Risk Probability", f"{risk_probability:.2%}", delta=None)
        
        with col3:
            st.metric("Recommendation", "✅ APPROVE" if risk_probability < 0.5 else "❌ REJECT")
        
        # Risk factors visualization
        st.subheader("📊 Risk Factor Analysis")
        
        factors = {
            'Credit Score Impact': 1 - (credit_score / 850),
            'Income-to-Loan Ratio': (loan_amount / income) if income > 0 else 0,
            'Debt-to-Income Ratio': (existing_debts / income) if income > 0 else 0,
            'Payment Ratio': monthly_payment_ratio,
            'Default History': previous_defaults / 5
        }
        
        factors_df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Risk Score'])
        
        fig = px.bar(
            factors_df, x='Factor', y='Risk Score',
            color='Risk Score',
            color_continuous_scale='RdYlGn_r',
            title="Normalized Risk Factors (0-1 scale)",
            height=400
        )
        fig.update_layout(showlegend=False, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: ANALYTICS (Portfolio Overview)
# ============================================================================

with tab2:
    st.header("Portfolio Analytics & Risk Distribution")
    
    # Get predictions for all data
    X_all = data[FEATURE_NAMES]
    X_scaled = scaler.transform(X_all)
    data['predicted_risk'] = model.predict_proba(X_scaled)[:, 1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", len(data))
    with col2:
        low_risk = (data['predicted_risk'] < 0.3).sum()
        st.metric("Low Risk", low_risk, f"{low_risk/len(data):.1%}")
    with col3:
        med_risk = ((data['predicted_risk'] >= 0.3) & (data['predicted_risk'] < 0.6)).sum()
        st.metric("Medium Risk", med_risk, f"{med_risk/len(data):.1%}")
    with col4:
        high_risk = (data['predicted_risk'] >= 0.6).sum()
        st.metric("High Risk", high_risk, f"{high_risk/len(data):.1%}")
    
    st.markdown("---")
    
    # Risk distribution histogram
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            data, x='predicted_risk', nbins=30,
            title="Risk Distribution Across Portfolio",
            labels={'predicted_risk': 'Risk Probability'},
            color_discrete_sequence=['#3498db']
        )
        fig.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
        fig.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="High Risk")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        risk_counts = pd.cut(
            data['predicted_risk'], 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        ).value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=['#27AE60', '#F39C12', '#E74C3C'])
        )])
        fig.update_layout(title="Risk Segmentation")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlations")
    corr_data = data[FEATURE_NAMES + ['predicted_risk']].corr()
    
    fig = px.imshow(
        corr_data,
        color_continuous_scale='RdBu',
        text_auto='.2f',
        title="Feature Correlation Matrix",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: CUSTOMER PROFILES
# ============================================================================

with tab3:
    st.header("Customer Risk Profiles")
    
    X_all = data[FEATURE_NAMES]
    X_scaled = scaler.transform(X_all)
    data['predicted_risk'] = model.predict_proba(X_scaled)[:, 1]
    
    # Filter by risk level
    risk_filter = st.selectbox("Filter by Risk Level", ["All", "Low", "Medium", "High"])
    
    if risk_filter == "Low":
        filtered = data[data['predicted_risk'] < 0.3]
    elif risk_filter == "Medium":
        filtered = data[(data['predicted_risk'] >= 0.3) & (data['predicted_risk'] < 0.6)]
    elif risk_filter == "High":
        filtered = data[data['predicted_risk'] >= 0.6]
    else:
        filtered = data
    
    # Display table with sorting
    st.subheader(f"Showing {len(filtered)} Customers ({risk_filter})")
    
    display_cols = ['age', 'income', 'credit_score', 'loan_amount', 'predicted_risk']
    display_data = filtered[display_cols].copy()
    display_data['predicted_risk'] = display_data['predicted_risk'].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(
        display_data.sort_values('age'),
        use_container_width=True,
        height=400
    )
    
    # Scatter plot: Income vs Credit Score colored by risk
    st.subheader("Income vs Credit Score Analysis")
    
    fig = px.scatter(
        filtered, x='income', y='credit_score',
        size='loan_amount',
        color='predicted_risk',
        hover_data=['age', 'employment_years'],
        color_continuous_scale='RdYlGn_r',
        title="Customer Distribution",
        labels={'predicted_risk': 'Risk Probability'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: MODEL INSIGHTS
# ============================================================================

with tab4:
    st.header("🎯 Model Performance & Feature Importance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "XGBoost Classifier")
    with col2:
        st.metric("Training Samples", len(data))
    with col3:
        st.metric("Features Used", len(FEATURE_NAMES))
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("📊 Feature Importance")
    
    importance_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        title="Which Features Influence Risk Predictions Most?"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model info
    st.subheader("ℹ️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**XGBoost Parameters:**")
        for param, value in MODEL_PARAMS.items():
            st.write(f"- {param}: {value}")
    
    with col2:
        st.write("**Risk Thresholds:**")
        for level, params in RISK_LEVELS.items():
            st.write(f"- {level}: {params['range'][0]:.1%} - {params['range'][1]:.1%}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Credit Risk Assessment Dashboard v1.0 | Data Science Masters Project</p>",
    unsafe_allow_html=True
)
