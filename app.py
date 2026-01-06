import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openai import OpenAI
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="Universal DQS Prototype", layout="wide")
client = OpenAI(api_key=st.sidebar.text_input("OpenAI API Key", type="password"))

# --- CORE SCORING ENGINE (Deterministic) ---
def calculate_metrics(df):
    total_rows = len(df)
    results = {}
    
    # 1. Completeness: % of non-null cells
    results['Completeness'] = (df.notnull().sum().sum() / (total_rows * len(df.columns)))
    
    # 2. Uniqueness: % of unique rows
    results['Uniqueness'] = len(df.drop_duplicates()) / total_rows
    
    # 3. Validity: Payments Specific (Amount > 0)
    if 'amount' in df.columns:
        valid_amt = (df['amount'] > 0).sum() / total_rows
        results['Validity'] = valid_amt
    else:
        results['Validity'] = 1.0 # Default if column missing
        
    # 4. Consistency: Example - Currency code length
    if 'currency' in df.columns:
        consistent_cur = (df['currency'].astype(str).str.len() == 3).sum() / total_rows
        results['Consistency'] = consistent_cur
    else:
        results['Consistency'] = 1.0

    # Composite Score (Weighted)
    weights = {'Completeness': 0.3, 'Uniqueness': 0.2, 'Validity': 0.3, 'Consistency': 0.2}
    composite = sum(results[k] * weights.get(k, 0) for k in results) * 100
    
    return results, composite

# --- GENAI INSIGHT LAYER ---
def get_ai_insights(metrics, composite_score, schema):
    prompt = f"""
    You are a Senior Data Quality Auditor in the Payments industry. 
    Analyze these results and provide a 3-sentence executive summary.
    
    METRICS: {metrics}
    OVERALL DQS: {composite_score:.2f}/100
    SCHEMA: {list(schema.keys())}

    Format your response as:
    ### AI Executive Summary
    [Summary here focusing on business risk]
    ### Prioritized Recommendations
    - [High Priority Fix]
    - [Process Improvement]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You provide technical data quality audits."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Connect your API key to see AI insights."

# --- UI LAYER (Streamlit) ---
st.title("💳 Payments Data Quality Scoring (DQS)")
st.markdown("### Hackathon Prototype: Universal Dimension-Based Assessment")

uploaded_file = st.file_uploader("Upload Transaction Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    metrics, dqs = calculate_metrics(df)
    
    # Header Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Composite DQS", f"{dqs:.1f}%")
    c2.metric("Total Transactions", len(df))
    status = "EXCELLENT" if dqs > 90 else "ACCEPTABLE" if dqs > 75 else "CRITICAL"
    c3.metric("Status", status)

    # Visualization: Radar Chart
    df_radar = pd.DataFrame(dict(r=list(metrics.values()), theta=list(metrics.keys())))
    fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True, range_r=[0,1])
    
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        # AI Explainability
        schema_info = df.dtypes.to_dict()
        with st.spinner("Generating AI Insights..."):
            insights = get_ai_insights(metrics, dqs, schema_info)
            st.markdown(insights)

    # Detailed Table
    with st.expander("View Dimension Breakdown"):
        st.table(pd.DataFrame([metrics]).T.rename(columns={0: "Score (0.0-1.0)"}))