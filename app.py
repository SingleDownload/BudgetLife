import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score, r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import label_binarize
from mlxtend.frequent_patterns import apriori, association_rules
import io
import json

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BudgetLife Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0D3B5E;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #1B5E8C;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1B5E8C 0%, #0D3B5E 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(13, 59, 94, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    .insight-box {
        background: #F0F8FF;
        border-left: 4px solid #1B5E8C;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .strategy-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
        border-left: 4px solid #27AE60;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .warning-box {
        background: #FFF8E1;
        border-left: 4px solid #F9A825;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    div[data-testid="stMetric"] {
        background: #F7FBFE;
        border: 1px solid #E0E0E0;
        padding: 0.8rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

COLORS = ["#1B5E8C", "#27AE60", "#E67E22", "#8E44AD", "#E74C3C", "#16A085", "#2C3E50", "#F39C12"]

# ═══════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv("BudgetLife_Synthetic_Dataset.csv")
    return df

@st.cache_data
def preprocess_for_ml(df):
    ordinal_maps = {
        "Q1_Age_Group": ["18–24", "25–34", "35–44", "45–54", "55+"],
        "Q3_City_Tier": ["Rural", "Tier 3", "Tier 2", "Metro (Tier 1)"],
        "Q4_Financial_Dependents": ["0", "1", "2", "3", "4+"],
        "Q6_Monthly_Income": ["Below ₹20,000", "₹20,001–₹50,000", "₹50,001–₹1,00,000", "₹1,00,001–₹2,00,000", "Above ₹2,00,000"],
        "Q7_Income_Stability": ["Highly unpredictable", "Irregular but manageable", "Mostly stable with some variation", "Very stable (fixed salary)"],
        "Q8_Monthly_Expenditure": ["Below ₹10,000", "₹10,001–₹25,000", "₹25,001–₹50,000", "₹50,001–₹1,00,000", "Above ₹1,00,000"],
        "Q10_Impulse_Purchase_Frequency": ["Never", "Rarely (1–2 times/month)", "Sometimes (3–5 times/month)", "Often (6+ times/month)"],
        "Q12_Active_Subscriptions": ["None", "1–2", "3–5", "6+"],
        "Q13_Statement_Review_Frequency": ["Never", "Rarely", "Quarterly", "Monthly", "Weekly"],
        "Q15_Savings_Percentage": ["0% (No savings)", "1–10%", "11–20%", "21–30%", "Above 30%"],
        "Q17_Financial_Confidence": ["1 – Not confident at all", "2 – Slightly confident", "3 – Moderately confident", "4 – Very confident", "5 – Extremely confident"],
        "Q18_Financial_Stress_Level": ["1 – Never", "2 – Rarely", "3 – Sometimes", "4 – Often", "5 – Almost always"],
        "Q21_Financial_Literacy": ["1 – Very Low", "2 – Low", "3 – Average", "4 – High", "5 – Very High"],
        "Q24_Willingness_To_Pay": ["₹0 (Free only)", "₹49–₹99", "₹100–₹199", "₹200–₹499", "₹500+"],
    }
    nominal_cols = ["Q2_Marital_Dependent_Status", "Q5_Employment_Status", "Q19_Budget_Behavior",
                    "Q20_Finance_App_Usage", "Q22_Digital_Comfort_Level"]

    df_enc = df.drop(columns=["Respondent_ID", "Persona_Tag"], errors="ignore").copy()

    multi_cols = ["Q9_Top_Spending_Categories", "Q11_Spending_Triggers", "Q14_Financial_Goals",
                  "Q16_Financial_Challenges", "Q23_Preferred_Features"]
    for col in multi_cols:
        if col in df_enc.columns:
            unique_vals = set()
            df_enc[col].dropna().apply(lambda x: unique_vals.update(x.split("|")))
            for val in unique_vals:
                safe_name = f"{col}__{val.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('/', '_')}"
                df_enc[safe_name] = df_enc[col].apply(lambda x: 1 if val in str(x).split("|") else 0)
            df_enc.drop(columns=[col], inplace=True)

    for col, order in ordinal_maps.items():
        if col in df_enc.columns:
            mapping = {v: i for i, v in enumerate(order)}
            df_enc[col] = df_enc[col].map(mapping).fillna(0).astype(int)

    for col in nominal_cols:
        if col in df_enc.columns:
            dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=True)
            df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)

    if "Q25_BudgetLife_Interest" in df_enc.columns:
        target_map = {
            "Definitely Yes": "Likely Adopter", "Probably Yes": "Likely Adopter",
            "Maybe": "Persuadable",
            "Probably No": "Unlikely", "Definitely No": "Unlikely"
        }
        df_enc["Target"] = df_enc["Q25_BudgetLife_Interest"].map(target_map)
        df_enc.drop(columns=["Q25_BudgetLife_Interest"], inplace=True)

    return df_enc, ordinal_maps

def preprocess_new_customer(new_df, ordinal_maps, train_columns):
    df_enc = new_df.copy()
    multi_cols = ["Q9_Top_Spending_Categories", "Q11_Spending_Triggers", "Q14_Financial_Goals",
                  "Q16_Financial_Challenges", "Q23_Preferred_Features"]
    for col in multi_cols:
        if col in df_enc.columns:
            unique_vals = set()
            df_enc[col].dropna().apply(lambda x: unique_vals.update(str(x).split("|")))
            for val in unique_vals:
                safe_name = f"{col}__{val.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('/', '_')}"
                df_enc[safe_name] = df_enc[col].apply(lambda x: 1 if val in str(x).split("|") else 0)
            df_enc.drop(columns=[col], inplace=True)

    nominal_cols = ["Q2_Marital_Dependent_Status", "Q5_Employment_Status", "Q19_Budget_Behavior",
                    "Q20_Finance_App_Usage", "Q22_Digital_Comfort_Level"]

    for col, order in ordinal_maps.items():
        if col in df_enc.columns:
            mapping = {v: i for i, v in enumerate(order)}
            df_enc[col] = df_enc[col].map(mapping).fillna(0).astype(int)

    for col in nominal_cols:
        if col in df_enc.columns:
            dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=True)
            df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)

    if "Q25_BudgetLife_Interest" in df_enc.columns:
        df_enc.drop(columns=["Q25_BudgetLife_Interest"], inplace=True)

    for col in train_columns:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[train_columns]
    return df_enc

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file 'BudgetLife_Synthetic_Dataset.csv' not found. Please ensure it is in the same directory as app.py.")
    st.stop()

df_ml, ordinal_maps = preprocess_for_ml(df)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 💰 BudgetLife")
    st.markdown("**Data-Driven Decision Engine**")
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df):,} respondents")
    st.markdown(f"**Features:** 25 survey questions")
    st.markdown(f"**Personas:** {df['Persona_Tag'].nunique()} segments")
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    tab_choice = st.radio(
        "Select Analysis Module",
        ["📊 Overview & Descriptive", "🔬 Clustering Analysis", "🔗 Association Rule Mining",
         "🎯 Classification", "📈 Regression Analysis", "💡 Prescriptive Strategy",
         "🔮 New Customer Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("##### Built for BudgetLife")
    st.caption("Smart Expense Optimization Platform")

# ═══════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW & DESCRIPTIVE
# ═══════════════════════════════════════════════════════════════

if tab_choice == "📊 Overview & Descriptive":
    st.markdown('<div class="main-header">📊 Descriptive Analysis — Market Landscape</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding who our potential customers are and how they manage money</div>', unsafe_allow_html=True)

    # KPI Metrics
    likely = df["Q25_BudgetLife_Interest"].isin(["Definitely Yes", "Probably Yes"]).sum()
    avg_stress = df["Q18_Financial_Stress_Level"].map({"1 – Never": 1, "2 – Rarely": 2, "3 – Sometimes": 3, "4 – Often": 4, "5 – Almost always": 5}).mean()
    no_budget = df["Q19_Budget_Behavior"].isin(["No, but I want to", "No, and I don't plan to"]).sum()
    never_app = (df["Q20_Finance_App_Usage"] == "No, never used one").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Respondents</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{likely/len(df)*100:.1f}%</div><div class="metric-label">Likely Adopters</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_stress:.2f}/5</div><div class="metric-label">Avg Financial Stress</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{no_budget/len(df)*100:.1f}%</div><div class="metric-label">No Budget in Place</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Demographics
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Q1_Age_Group", color="Q1_Age_Group", color_discrete_sequence=COLORS,
                          title="Age Distribution", category_orders={"Q1_Age_Group": ["18–24", "25–34", "35–44", "45–54", "55+"]})
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(df, names="Q3_City_Tier", title="City Tier Distribution", color_discrete_sequence=COLORS, hole=0.4)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Q6_Monthly_Income", color="Q6_Monthly_Income", color_discrete_sequence=COLORS,
                          title="Monthly Income Distribution",
                          category_orders={"Q6_Monthly_Income": ["Below ₹20,000", "₹20,001–₹50,000", "₹50,001–₹1,00,000", "₹1,00,001–₹2,00,000", "Above ₹2,00,000"]})
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="Q15_Savings_Percentage", color="Q15_Savings_Percentage", color_discrete_sequence=COLORS,
                          title="Savings Rate Distribution",
                          category_orders={"Q15_Savings_Percentage": ["0% (No savings)", "1–10%", "11–20%", "21–30%", "Above 30%"]})
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    # Spending Categories
    st.markdown("### 🛒 Top Spending Categories (Multi-Select Analysis)")
    spending_cats = df["Q9_Top_Spending_Categories"].str.split("|").explode().value_counts()
    fig = px.bar(x=spending_cats.values, y=spending_cats.index, orientation="h", color=spending_cats.values,
                color_continuous_scale="Teal", title="Most Common Spending Categories Across All Respondents")
    fig.update_layout(height=420, yaxis_title="", xaxis_title="Frequency", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Financial Health Heatmap
    st.markdown("### 🧠 Financial Confidence vs. Stress Heatmap")
    cross = pd.crosstab(df["Q17_Financial_Confidence"], df["Q18_Financial_Stress_Level"])
    fig = px.imshow(cross, text_auto=True, color_continuous_scale="Blues",
                    title="Financial Confidence × Financial Stress Cross-Tabulation")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # BudgetLife Interest by Persona
    st.markdown("### 🎯 BudgetLife Interest by Persona")
    interest_persona = pd.crosstab(df["Persona_Tag"], df["Q25_BudgetLife_Interest"], normalize="index") * 100
    interest_order = ["Definitely Yes", "Probably Yes", "Maybe", "Probably No", "Definitely No"]
    interest_persona = interest_persona.reindex(columns=[c for c in interest_order if c in interest_persona.columns])
    fig = px.bar(interest_persona, barmode="stack", color_discrete_sequence=["#27AE60", "#82E0AA", "#F9E79F", "#E67E22", "#E74C3C"],
                title="Adoption Interest Distribution by Persona (%)")
    fig.update_layout(height=420, yaxis_title="Percentage (%)", xaxis_title="", xaxis_tickangle=-20, legend_title="Interest Level")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">📌 <b>Key Insight:</b> Over half the respondents lack a structured budget, yet the majority express moderate-to-high financial stress — this gap represents BudgetLife\'s core market opportunity.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: CLUSTERING
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "🔬 Clustering Analysis":
    st.markdown('<div class="main-header">🔬 Diagnostic Analysis — Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discovering natural customer segments using K-Means Clustering</div>', unsafe_allow_html=True)

    cluster_features = [
        "Q1_Age_Group", "Q3_City_Tier", "Q4_Financial_Dependents", "Q6_Monthly_Income",
        "Q7_Income_Stability", "Q8_Monthly_Expenditure", "Q10_Impulse_Purchase_Frequency",
        "Q12_Active_Subscriptions", "Q13_Statement_Review_Frequency", "Q15_Savings_Percentage",
        "Q17_Financial_Confidence", "Q18_Financial_Stress_Level", "Q21_Financial_Literacy",
        "Q24_Willingness_To_Pay"
    ]

    df_clust = df_ml[[c for c in cluster_features if c in df_ml.columns]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clust)

    # Elbow + Silhouette
    st.markdown("### Optimal Cluster Selection")
    col1, col2 = st.columns(2)

    inertias, sil_scores = [], []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode="lines+markers", marker=dict(size=10, color="#1B5E8C"),
                                 line=dict(width=3, color="#1B5E8C")))
        fig.update_layout(title="Elbow Method — Inertia vs. K", xaxis_title="Number of Clusters (K)", yaxis_title="Inertia", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=sil_scores, mode="lines+markers", marker=dict(size=10, color="#27AE60"),
                                 line=dict(width=3, color="#27AE60")))
        fig.update_layout(title="Silhouette Score vs. K", xaxis_title="Number of Clusters (K)", yaxis_title="Silhouette Score", height=380)
        st.plotly_chart(fig, use_container_width=True)

    optimal_k = list(K_range)[np.argmax(sil_scores)]
    st.markdown(f'<div class="insight-box">📌 <b>Optimal K = {optimal_k}</b> — Highest silhouette score of {max(sil_scores):.4f}, indicating the best-separated cluster structure.</div>', unsafe_allow_html=True)

    # Fit final model
    n_clusters = st.slider("Select number of clusters for analysis", 2, 8, optimal_k)
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = km_final.fit_predict(X_scaled)

    # Cluster Size
    st.markdown("### Cluster Distribution")
    cluster_counts = df["Cluster"].value_counts().sort_index()
    fig = px.bar(x=[f"Cluster {i}" for i in cluster_counts.index], y=cluster_counts.values,
                color=[f"Cluster {i}" for i in cluster_counts.index], color_discrete_sequence=COLORS,
                title="Respondents per Cluster")
    fig.update_layout(showlegend=False, height=350, xaxis_title="", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # Cluster Profiles — Radar
    st.markdown("### Cluster Profile Comparison (Radar Chart)")
    radar_features = ["Q6_Monthly_Income", "Q8_Monthly_Expenditure", "Q15_Savings_Percentage",
                      "Q10_Impulse_Purchase_Frequency", "Q17_Financial_Confidence",
                      "Q18_Financial_Stress_Level", "Q21_Financial_Literacy", "Q24_Willingness_To_Pay"]
    radar_labels = ["Income", "Expenditure", "Savings", "Impulse Spending", "Confidence", "Stress", "Literacy", "WTP"]

    radar_df = df_ml.copy()
    radar_df["Cluster"] = df["Cluster"]
    cluster_means = radar_df.groupby("Cluster")[radar_features].mean()

    # Normalize to 0-1 for radar
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    cluster_means_norm = pd.DataFrame(mms.fit_transform(cluster_means), columns=radar_features, index=cluster_means.index)

    fig = go.Figure()
    for i in range(n_clusters):
        values = cluster_means_norm.loc[i].tolist()
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(r=values, theta=radar_labels + [radar_labels[0]],
                                       fill="toself", name=f"Cluster {i}", line=dict(color=COLORS[i % len(COLORS)])))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Cluster Profiles — Normalized Feature Comparison", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Cluster x BudgetLife Interest
    st.markdown("### Cluster × BudgetLife Adoption Interest")
    cluster_interest = pd.crosstab(df["Cluster"], df["Q25_BudgetLife_Interest"], normalize="index") * 100
    interest_order = ["Definitely Yes", "Probably Yes", "Maybe", "Probably No", "Definitely No"]
    cluster_interest = cluster_interest.reindex(columns=[c for c in interest_order if c in cluster_interest.columns])
    cluster_interest.index = [f"Cluster {i}" for i in cluster_interest.index]
    fig = px.bar(cluster_interest, barmode="stack", color_discrete_sequence=["#27AE60", "#82E0AA", "#F9E79F", "#E67E22", "#E74C3C"],
                title="Adoption Likelihood by Cluster")
    fig.update_layout(height=400, yaxis_title="Percentage (%)", xaxis_title="", legend_title="Interest Level")
    st.plotly_chart(fig, use_container_width=True)

    # Cluster Summary Table
    st.markdown("### Cluster Summary Statistics")
    summary_cols = {"Q6_Monthly_Income": "Avg Income", "Q8_Monthly_Expenditure": "Avg Expenditure",
                    "Q15_Savings_Percentage": "Avg Savings", "Q18_Financial_Stress_Level": "Avg Stress",
                    "Q21_Financial_Literacy": "Avg Literacy", "Q24_Willingness_To_Pay": "Avg WTP"}
    summary = radar_df.groupby("Cluster")[list(summary_cols.keys())].mean().round(2)
    summary.columns = list(summary_cols.values())
    summary["Size"] = cluster_counts.values
    summary.index = [f"Cluster {i}" for i in summary.index]
    st.dataframe(summary, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: ASSOCIATION RULE MINING
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "🔗 Association Rule Mining":
    st.markdown('<div class="main-header">🔗 Diagnostic Analysis — Association Rule Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discovering hidden relationships between spending patterns, challenges, and feature preferences</div>', unsafe_allow_html=True)

    # Build transaction matrix
    multi_cols = {
        "Q9_Top_Spending_Categories": "Spend",
        "Q11_Spending_Triggers": "Trigger",
        "Q14_Financial_Goals": "Goal",
        "Q16_Financial_Challenges": "Challenge",
        "Q23_Preferred_Features": "Feature"
    }

    transactions = pd.DataFrame()
    for col, prefix in multi_cols.items():
        items = df[col].str.get_dummies(sep="|")
        items.columns = [f"{prefix}: {c}" for c in items.columns]
        transactions = pd.concat([transactions, items], axis=1)

    # Add key single-choice items as binary
    transactions["High Impulse Spender"] = (df["Q10_Impulse_Purchase_Frequency"].isin(["Sometimes (3–5 times/month)", "Often (6+ times/month)"])).astype(int)
    transactions["Low Saver (0-10%)"] = (df["Q15_Savings_Percentage"].isin(["0% (No savings)", "1–10%"])).astype(int)
    transactions["No Budget"] = (df["Q19_Budget_Behavior"].isin(["No, but I want to", "No, and I don't plan to"])).astype(int)
    transactions["Never Used Finance App"] = (df["Q20_Finance_App_Usage"] == "No, never used one").astype(int)
    transactions["High Financial Stress"] = (df["Q18_Financial_Stress_Level"].isin(["4 – Often", "5 – Almost always"])).astype(int)

    min_support = st.slider("Minimum Support Threshold", 0.03, 0.20, 0.07, 0.01)
    min_confidence = st.slider("Minimum Confidence Threshold", 0.30, 0.80, 0.40, 0.05)

    frequent_items = apriori(transactions, min_support=min_support, use_colnames=True)

    if len(frequent_items) == 0:
        st.warning("No frequent itemsets found. Try lowering the support threshold.")
    else:
        rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_items))
        rules = rules[rules["lift"] > 1.0]
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

        if len(rules) == 0:
            st.warning("No rules found with current thresholds. Try adjusting the sliders.")
        else:
            st.markdown(f"### Discovered {len(rules)} Association Rules")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rules", len(rules))
            with col2:
                st.metric("Avg Confidence", f"{rules['confidence'].mean():.2f}")
            with col3:
                st.metric("Avg Lift", f"{rules['lift'].mean():.2f}")

            # Top Rules Table
            st.markdown("### Top 20 Rules by Lift")
            top_rules = rules.nlargest(20, "lift")[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)
            top_rules.columns = ["If (Antecedent)", "Then (Consequent)", "Support", "Confidence", "Lift"]
            top_rules["Support"] = top_rules["Support"].round(3)
            top_rules["Confidence"] = top_rules["Confidence"].round(3)
            top_rules["Lift"] = top_rules["Lift"].round(3)
            st.dataframe(top_rules, use_container_width=True, height=500)

            # Scatter: Confidence vs Lift
            st.markdown("### Confidence vs. Lift Scatter Plot")
            fig = px.scatter(rules, x="confidence", y="lift", size="support", color="lift",
                           color_continuous_scale="Viridis", hover_data=["antecedents", "consequents"],
                           title="Association Rules — Confidence vs. Lift (size = Support)")
            fig.update_layout(height=480, xaxis_title="Confidence", yaxis_title="Lift")
            st.plotly_chart(fig, use_container_width=True)

            # Support vs Confidence
            st.markdown("### Support vs. Confidence Scatter Plot")
            fig = px.scatter(rules, x="support", y="confidence", size="lift", color="confidence",
                           color_continuous_scale="Teal", hover_data=["antecedents", "consequents"],
                           title="Association Rules — Support vs. Confidence (size = Lift)")
            fig.update_layout(height=480, xaxis_title="Support", yaxis_title="Confidence")
            st.plotly_chart(fig, use_container_width=True)

            # Top rules by confidence
            st.markdown("### Top 15 Rules by Confidence")
            top_conf = rules.nlargest(15, "confidence")
            fig = px.bar(top_conf, x="confidence", y=top_conf["antecedents"].str[:50] + " → " + top_conf["consequents"].str[:50],
                        orientation="h", color="lift", color_continuous_scale="Oranges",
                        title="Highest Confidence Rules (color = Lift)")
            fig.update_layout(height=550, yaxis_title="", xaxis_title="Confidence")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="insight-box">📌 <b>Business Insight:</b> Rules with high confidence AND high lift reveal which behavioral combinations most strongly predict interest in specific BudgetLife features — use these to personalize onboarding and marketing.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4: CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "🎯 Classification":
    st.markdown('<div class="main-header">🎯 Predictive Analysis — Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicting which customers are most likely to adopt BudgetLife</div>', unsafe_allow_html=True)

    X = df_ml.drop(columns=["Target"], errors="ignore")
    y = df_ml["Target"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Models
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    lr = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial")

    rf.fit(X_train_sc, y_train)
    lr.fit(X_train_sc, y_train)

    rf_pred = rf.predict(X_test_sc)
    lr_pred = lr.predict(X_test_sc)

    st.markdown("### Model Performance Comparison")
    col1, col2 = st.columns(2)

    for col, model_name, y_pred, model in [(col1, "Random Forest", rf_pred, rf), (col2, "Logistic Regression", lr_pred, lr)]:
        with col:
            st.markdown(f"#### {model_name}")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("Precision", f"{prec:.4f}")
            m3, m4 = st.columns(2)
            m3.metric("Recall", f"{rec:.4f}")
            m4.metric("F1-Score", f"{f1:.4f}")

    # Classification Reports
    st.markdown("### Detailed Classification Report")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        report_rf = classification_report(y_test, rf_pred, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report_rf).T.round(4), use_container_width=True)
    with col2:
        st.markdown("**Logistic Regression**")
        report_lr = classification_report(y_test, lr_pred, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report_lr).T.round(4), use_container_width=True)

    # Confusion Matrices
    st.markdown("### Confusion Matrices")
    col1, col2 = st.columns(2)
    for col_obj, name, preds in [(col1, "Random Forest", rf_pred), (col2, "Logistic Regression", lr_pred)]:
        with col_obj:
            cm = confusion_matrix(y_test, preds)
            fig = px.imshow(cm, text_auto=True, x=class_names, y=class_names, color_continuous_scale="Blues",
                          title=f"{name} — Confusion Matrix")
            fig.update_layout(height=380, xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)

    # ROC Curves
    st.markdown("### ROC Curves (One-vs-Rest)")
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Random Forest ROC", "Logistic Regression ROC"])

    for idx, (model, model_name) in enumerate([(rf, "Random Forest"), (lr, "Logistic Regression")]):
        y_prob = model.predict_proba(X_test_sc)
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{cls} (AUC={roc_auc:.3f})",
                                     line=dict(color=COLORS[i % len(COLORS)]), showlegend=(idx == 0)),
                         row=1, col=idx + 1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"),
                                 showlegend=False), row=1, col=idx + 1)

    fig.update_layout(height=450, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.markdown("### Feature Importance — Random Forest (Top 20)")
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(20)
    fig = px.bar(x=feat_imp.values, y=feat_imp.index, orientation="h", color=feat_imp.values,
                color_continuous_scale="Teal", title="Top 20 Most Important Features for Adoption Prediction")
    fig.update_layout(height=550, yaxis_title="", xaxis_title="Importance", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">📌 <b>Key Finding:</b> The features driving adoption prediction reveal what BudgetLife should emphasize in marketing — focus messaging around the top importance drivers.</div>', unsafe_allow_html=True)

    # Store in session state
    st.session_state["rf_model"] = rf
    st.session_state["lr_model"] = lr
    st.session_state["scaler"] = scaler
    st.session_state["le"] = le
    st.session_state["train_columns"] = list(X.columns)


# ═══════════════════════════════════════════════════════════════
# TAB 5: REGRESSION
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "📈 Regression Analysis":
    st.markdown('<div class="main-header">📈 Predictive Analysis — Regression Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicting monthly expenditure, savings potential, and willingness to pay</div>', unsafe_allow_html=True)

    regression_targets = {
        "Monthly Expenditure": "Q8_Monthly_Expenditure",
        "Savings Potential": "Q15_Savings_Percentage",
        "Willingness to Pay": "Q24_Willingness_To_Pay"
    }

    reg_features = [c for c in df_ml.columns if c not in ["Target", "Q8_Monthly_Expenditure", "Q15_Savings_Percentage", "Q24_Willingness_To_Pay"]]

    results = {}
    for target_name, target_col in regression_targets.items():
        X_reg = df_ml[reg_features].copy()
        y_reg = df_ml[target_col].copy()

        X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_sc, y_tr)
        y_pred = ridge.predict(X_te_sc)

        results[target_name] = {
            "r2": r2_score(y_te, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
            "mae": mean_absolute_error(y_te, y_pred),
            "y_te": y_te, "y_pred": y_pred,
            "coefs": pd.Series(ridge.coef_, index=reg_features)
        }

    # Summary Metrics
    st.markdown("### Model Performance Summary")
    perf_data = []
    for name, res in results.items():
        perf_data.append({"Target": name, "R² Score": round(res["r2"], 4), "RMSE": round(res["rmse"], 4), "MAE": round(res["mae"], 4)})
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    # Actual vs Predicted
    st.markdown("### Actual vs. Predicted Plots")
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            fig = px.scatter(x=res["y_te"], y=res["y_pred"], title=f"{name}",
                           labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=[COLORS[i]])
            max_val = max(max(res["y_te"]), max(res["y_pred"]))
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                     line=dict(dash="dash", color="red"), showlegend=False))
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    # Residual Distributions
    st.markdown("### Residual Analysis")
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            residuals = res["y_te"].values - res["y_pred"]
            fig = px.histogram(x=residuals, nbins=30, title=f"{name} — Residuals",
                             color_discrete_sequence=[COLORS[i]])
            fig.update_layout(height=350, xaxis_title="Residual", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    # Top Coefficients
    st.markdown("### Top Feature Coefficients (Ridge Regression)")
    target_select = st.selectbox("Select Target Variable", list(results.keys()))
    coefs = results[target_select]["coefs"]
    top_coefs = pd.concat([coefs.nlargest(10), coefs.nsmallest(10)]).sort_values()
    fig = px.bar(x=top_coefs.values, y=top_coefs.index, orientation="h",
                color=top_coefs.values, color_continuous_scale="RdBu_r",
                title=f"Top Positive & Negative Predictors — {target_select}")
    fig.update_layout(height=550, yaxis_title="", xaxis_title="Coefficient", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">📌 <b>Business Application:</b> These regression models enable BudgetLife to estimate each user\'s spending, savings potential, and price sensitivity during onboarding — powering personalized budgets and dynamic pricing.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 6: PRESCRIPTIVE STRATEGY
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "💡 Prescriptive Strategy":
    st.markdown('<div class="main-header">💡 Prescriptive Analysis — Strategic Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Translating data insights into actionable business strategy for BudgetLife</div>', unsafe_allow_html=True)

    # Segment strategy
    st.markdown("### 🎯 Target Segment Prioritization")

    # Build segment analysis from personas
    persona_analysis = []
    for persona in df["Persona_Tag"].unique():
        p_df = df[df["Persona_Tag"] == persona]
        likely_pct = p_df["Q25_BudgetLife_Interest"].isin(["Definitely Yes", "Probably Yes"]).mean() * 100
        wtp_map = {"₹0 (Free only)": 0, "₹49–₹99": 74, "₹100–₹199": 150, "₹200–₹499": 350, "₹500+": 500}
        avg_wtp = p_df["Q24_Willingness_To_Pay"].map(wtp_map).mean()
        stress_map = {"1 – Never": 1, "2 – Rarely": 2, "3 – Sometimes": 3, "4 – Often": 4, "5 – Almost always": 5}
        avg_stress = p_df["Q18_Financial_Stress_Level"].map(stress_map).mean()
        digital_comfort_pct = p_df["Q22_Digital_Comfort_Level"].isin(["Very comfortable", "Somewhat comfortable"]).mean() * 100

        persona_analysis.append({
            "Segment": persona, "Size": len(p_df), "Adoption Likelihood (%)": round(likely_pct, 1),
            "Avg WTP (₹)": round(avg_wtp), "Avg Stress": round(avg_stress, 2),
            "Digital Comfort (%)": round(digital_comfort_pct, 1),
            "Revenue Score": round(likely_pct * avg_wtp / 100, 1)
        })

    pa_df = pd.DataFrame(persona_analysis).sort_values("Revenue Score", ascending=False)
    st.dataframe(pa_df, use_container_width=True, hide_index=True)

    # Revenue Score visualization
    fig = px.bar(pa_df, x="Segment", y="Revenue Score", color="Adoption Likelihood (%)",
                color_continuous_scale="Greens", title="Segment Revenue Score (Adoption % × WTP)")
    fig.update_layout(height=420, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    # Adoption vs WTP Bubble
    st.markdown("### Adoption Likelihood vs. Willingness to Pay")
    fig = px.scatter(pa_df, x="Adoption Likelihood (%)", y="Avg WTP (₹)", size="Size",
                    color="Segment", color_discrete_sequence=COLORS, text="Segment",
                    title="Segment Positioning — Where to Focus Marketing")
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=480)
    fig.add_hline(y=pa_df["Avg WTP (₹)"].median(), line_dash="dash", line_color="gray", annotation_text="Median WTP")
    fig.add_vline(x=pa_df["Adoption Likelihood (%)"].median(), line_dash="dash", line_color="gray", annotation_text="Median Adoption")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="strategy-box">
    <b>🟢 Top-Right Quadrant (High Adoption + High WTP):</b> Primary target — invest heavily in acquisition. These are your paying early adopters.
    </div>
    <div class="insight-box">
    <b>🔵 Top-Left Quadrant (Low Adoption + High WTP):</b> Trust-building needed. These users can pay but need convincing — focus on credibility, security messaging, and referrals.
    </div>
    <div class="warning-box">
    <b>🟡 Bottom-Right Quadrant (High Adoption + Low WTP):</b> Freemium funnel candidates. They want the product but won't pay much — convert through free tier, upsell later.
    </div>
    """, unsafe_allow_html=True)

    # Feature Priority
    st.markdown("### 📦 Feature Demand Analysis")
    features_exploded = df["Q23_Preferred_Features"].str.split("|").explode()
    feat_demand = features_exploded.value_counts()
    fig = px.bar(x=feat_demand.values, y=feat_demand.index, orientation="h", color=feat_demand.values,
                color_continuous_scale="Teal", title="Feature Demand — What Customers Want Most")
    fig.update_layout(height=350, yaxis_title="", xaxis_title="Selection Count", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Feature demand by top segments
    st.markdown("### Feature Demand by Top 3 Revenue Segments")
    top3 = pa_df.nlargest(3, "Revenue Score")["Segment"].tolist()
    feat_by_seg = []
    for seg in top3:
        seg_feats = df[df["Persona_Tag"] == seg]["Q23_Preferred_Features"].str.split("|").explode().value_counts()
        for feat, count in seg_feats.items():
            feat_by_seg.append({"Segment": seg, "Feature": feat, "Count": count})
    feat_seg_df = pd.DataFrame(feat_by_seg)
    fig = px.bar(feat_seg_df, x="Feature", y="Count", color="Segment", barmode="group",
                color_discrete_sequence=COLORS, title="Feature Preferences of Top Revenue Segments")
    fig.update_layout(height=420, xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)

    # Pricing Strategy
    st.markdown("### 💰 Pricing Strategy Recommendation")
    wtp_dist = df["Q24_Willingness_To_Pay"].value_counts()
    wtp_order = ["₹0 (Free only)", "₹49–₹99", "₹100–₹199", "₹200–₹499", "₹500+"]
    wtp_dist = wtp_dist.reindex(wtp_order)
    fig = px.bar(x=wtp_order, y=wtp_dist.values, color=wtp_dist.values,
                color_continuous_scale="Oranges", title="Willingness to Pay Distribution")
    fig.update_layout(height=380, xaxis_title="Price Tier", yaxis_title="Respondents", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    free_pct = (df["Q24_Willingness_To_Pay"] == "₹0 (Free only)").mean() * 100
    mid_pct = df["Q24_Willingness_To_Pay"].isin(["₹49–₹99", "₹100–₹199"]).mean() * 100
    st.markdown(f"""
    <div class="strategy-box">
    <b>Recommended Pricing Model — Freemium + Premium Tiers:</b><br>
    • <b>Free Tier:</b> Basic expense tracking & categorization — captures the {free_pct:.0f}% who won't pay initially<br>
    • <b>Plus Tier (₹99/month):</b> Subscription detection + smart alerts — sweet spot for the {mid_pct:.0f}% willing to pay ₹49–₹199<br>
    • <b>Pro Tier (₹249/month):</b> Full AI recommendations, health score, peer benchmarking — for high-value segments
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 7: NEW CUSTOMER PREDICTOR
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "🔮 New Customer Predictor":
    st.markdown('<div class="main-header">🔮 New Customer Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict adoption likelihood for new potential customers — individually or in bulk</div>', unsafe_allow_html=True)

    # Ensure models are trained
    if "rf_model" not in st.session_state:
        st.markdown('<div class="warning-box">⚠️ <b>Models not trained yet.</b> Please visit the <b>🎯 Classification</b> tab first to train the prediction models, then return here.</div>', unsafe_allow_html=True)
        st.stop()

    rf_model = st.session_state["rf_model"]
    scaler = st.session_state["scaler"]
    le = st.session_state["le"]
    train_columns = st.session_state["train_columns"]

    pred_mode = st.radio("Prediction Mode", ["📝 Single Customer Entry", "📁 Bulk CSV Upload"], horizontal=True)

    if pred_mode == "📝 Single Customer Entry":
        st.markdown("### Enter Customer Profile")

        with st.form("single_customer_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                q1 = st.selectbox("Q1: Age Group", ["18–24", "25–34", "35–44", "45–54", "55+"])
                q2 = st.selectbox("Q2: Marital & Dependent Status", ["Single, no dependents", "Single, with dependents", "Married, no children", "Married, with children", "Joint family (supporting elders)"])
                q3 = st.selectbox("Q3: City Tier", ["Metro (Tier 1)", "Tier 2", "Tier 3", "Rural"])
                q4 = st.selectbox("Q4: Financial Dependents", ["0", "1", "2", "3", "4+"])
                q5 = st.selectbox("Q5: Employment Status", ["Salaried (Private)", "Salaried (Government)", "Self-employed", "Freelancer or Gig Worker", "Student", "Unemployed", "Retired"])
                q6 = st.selectbox("Q6: Monthly Income", ["Below ₹20,000", "₹20,001–₹50,000", "₹50,001–₹1,00,000", "₹1,00,001–₹2,00,000", "Above ₹2,00,000"])
                q7 = st.selectbox("Q7: Income Stability", ["Very stable (fixed salary)", "Mostly stable with some variation", "Irregular but manageable", "Highly unpredictable"])
                q8 = st.selectbox("Q8: Monthly Expenditure", ["Below ₹10,000", "₹10,001–₹25,000", "₹25,001–₹50,000", "₹50,001–₹1,00,000", "Above ₹1,00,000"])

            with col2:
                q9 = st.multiselect("Q9: Top Spending Categories (up to 3)", ["Rent & Housing", "Groceries & Food", "Transportation", "Healthcare", "Education", "Entertainment & Dining Out", "Subscriptions (OTT, Gym, etc.)", "Shopping & Fashion", "EMIs & Loan Repayments", "Other"], max_selections=3)
                q10 = st.selectbox("Q10: Impulse Purchase Frequency", ["Never", "Rarely (1–2 times/month)", "Sometimes (3–5 times/month)", "Often (6+ times/month)"])
                q11 = st.multiselect("Q11: Spending Triggers (up to 2)", ["Emotional stress or boredom", "Social pressure or peer influence", "Flash sales, discounts & offers", "Celebrations or festivals", "Convenience (food delivery, cabs, etc.)", "Social media influence", "I rarely spend impulsively"], max_selections=2)
                q12 = st.selectbox("Q12: Active Subscriptions", ["None", "1–2", "3–5", "6+"])
                q13 = st.selectbox("Q13: Statement Review Frequency", ["Weekly", "Monthly", "Quarterly", "Rarely", "Never"])
                q14 = st.multiselect("Q14: Financial Goals (up to 3)", ["Build Emergency Fund", "Save for Vacation", "Pay Off Debt", "Invest & Grow Wealth", "Save for Major Purchase (Home, Car)", "Children's Education", "Retirement Planning", "No Specific Goal"], max_selections=3)

            with col3:
                q15 = st.selectbox("Q15: Savings Percentage", ["0% (No savings)", "1–10%", "11–20%", "21–30%", "Above 30%"])
                q16 = st.multiselect("Q16: Financial Challenges (up to 2)", ["Overspending", "Low or Irregular Income", "High EMIs or Debt", "Lack of Financial Knowledge", "No Budgeting Discipline", "Unexpected Expenses", "High Cost of Living"], max_selections=2)
                q17 = st.selectbox("Q17: Financial Confidence", ["1 – Not confident at all", "2 – Slightly confident", "3 – Moderately confident", "4 – Very confident", "5 – Extremely confident"])
                q18 = st.selectbox("Q18: Financial Stress Level", ["1 – Never", "2 – Rarely", "3 – Sometimes", "4 – Often", "5 – Almost always"])
                q19 = st.selectbox("Q19: Budget Behavior", ["Yes, strictly", "Yes, loosely", "No, but I want to", "No, and I don't plan to"])
                q20 = st.selectbox("Q20: Finance App Usage", ["Yes, currently using one", "Used before but stopped", "No, never used one"])
                q21 = st.selectbox("Q21: Financial Literacy", ["1 – Very Low", "2 – Low", "3 – Average", "4 – High", "5 – Very High"])
                q22 = st.selectbox("Q22: Digital Comfort Level", ["Very comfortable", "Somewhat comfortable", "Neutral", "Somewhat uncomfortable", "Very uncomfortable"])
                q23 = st.multiselect("Q23: Preferred Features (up to 3)", ["Automatic Expense Tracking & Smart Categorization", "Subscription & Money Leak Detection", "Predictive Budget Alerts & Spending Forecasts", "Financial Health Score with Peer Benchmarking", "AI Savings Recommendations & Goal Engine"], max_selections=3)
                q24 = st.selectbox("Q24: Willingness to Pay", ["₹0 (Free only)", "₹49–₹99", "₹100–₹199", "₹200–₹499", "₹500+"])

            submitted = st.form_submit_button("🔮 Predict Adoption Likelihood", use_container_width=True)

        if submitted:
            if not q9:
                q9 = ["Groceries & Food"]
            if not q11:
                q11 = ["I rarely spend impulsively"]
            if not q14:
                q14 = ["No Specific Goal"]
            if not q16:
                q16 = ["Unexpected Expenses"]
            if not q23:
                q23 = ["Automatic Expense Tracking & Smart Categorization"]

            new_row = {
                "Q1_Age_Group": q1, "Q2_Marital_Dependent_Status": q2, "Q3_City_Tier": q3,
                "Q4_Financial_Dependents": q4, "Q5_Employment_Status": q5, "Q6_Monthly_Income": q6,
                "Q7_Income_Stability": q7, "Q8_Monthly_Expenditure": q8,
                "Q9_Top_Spending_Categories": "|".join(q9), "Q10_Impulse_Purchase_Frequency": q10,
                "Q11_Spending_Triggers": "|".join(q11), "Q12_Active_Subscriptions": q12,
                "Q13_Statement_Review_Frequency": q13, "Q14_Financial_Goals": "|".join(q14),
                "Q15_Savings_Percentage": q15, "Q16_Financial_Challenges": "|".join(q16),
                "Q17_Financial_Confidence": q17, "Q18_Financial_Stress_Level": q18,
                "Q19_Budget_Behavior": q19, "Q20_Finance_App_Usage": q20,
                "Q21_Financial_Literacy": q21, "Q22_Digital_Comfort_Level": q22,
                "Q23_Preferred_Features": "|".join(q23), "Q24_Willingness_To_Pay": q24,
            }

            new_df = pd.DataFrame([new_row])
            X_new = preprocess_new_customer(new_df, ordinal_maps, train_columns)
            X_new_sc = scaler.transform(X_new)

            pred = rf_model.predict(X_new_sc)[0]
            proba = rf_model.predict_proba(X_new_sc)[0]
            pred_label = le.inverse_transform([pred])[0]

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            color_map = {"Likely Adopter": "#27AE60", "Persuadable": "#F9A825", "Unlikely": "#E74C3C"}
            emoji_map = {"Likely Adopter": "✅", "Persuadable": "🟡", "Unlikely": "❌"}

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color_map.get(pred_label, '#1B5E8C')}22, {color_map.get(pred_label, '#1B5E8C')}11);
                        border-left: 6px solid {color_map.get(pred_label, '#1B5E8C')}; padding: 1.5rem; border-radius: 0 12px 12px 0; margin: 1rem 0;">
                <h2 style="margin:0; color: {color_map.get(pred_label, '#1B5E8C')}">{emoji_map.get(pred_label, '')} {pred_label}</h2>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            col1, col2, col3 = st.columns(3)
            for i, cls in enumerate(le.classes_):
                col = [col1, col2, col3][i]
                with col:
                    st.metric(cls, f"{proba[i]*100:.1f}%")

            fig = px.bar(x=le.classes_, y=proba * 100, color=le.classes_,
                        color_discrete_map={"Likely Adopter": "#27AE60", "Persuadable": "#F9A825", "Unlikely": "#E74C3C"},
                        title="Prediction Confidence Breakdown")
            fig.update_layout(height=350, yaxis_title="Probability (%)", xaxis_title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.markdown("### 📋 Personalized Marketing Recommendation")
            if pred_label == "Likely Adopter":
                st.markdown("""
                <div class="strategy-box">
                <b>Strategy: Direct Conversion</b><br>
                • This customer is highly likely to adopt — push for immediate sign-up<br>
                • Offer a 14-day Premium trial to demonstrate full value<br>
                • Highlight AI Savings Recommendations and Predictive Alerts<br>
                • Channel: In-app onboarding + personalized email sequence
                </div>
                """, unsafe_allow_html=True)
            elif pred_label == "Persuadable":
                st.markdown("""
                <div class="warning-box">
                <b>Strategy: Nurture & Educate</b><br>
                • This customer needs more convincing — build trust first<br>
                • Send educational content about financial health and savings tips<br>
                • Offer free tier access with gentle upsell nudges<br>
                • Channel: Content marketing + social proof (testimonials, case studies)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-box">
                <b>Strategy: Low Priority / Long-term Nurture</b><br>
                • This customer is unlikely to convert now — avoid aggressive marketing<br>
                • Add to awareness campaigns and retarget in 3–6 months<br>
                • Focus on building brand recognition through thought leadership<br>
                • Channel: Low-cost social media ads + blog content
                </div>
                """, unsafe_allow_html=True)

    # ── BULK UPLOAD ──
    else:
        st.markdown("### Upload New Customer Data (CSV)")
        st.markdown("""
        Upload a CSV file with the same column structure as the survey dataset (Q1–Q24).
        The system will predict each customer's adoption likelihood and provide downloadable results.
        """)

        # Download template
        template_cols = [
            "Q1_Age_Group", "Q2_Marital_Dependent_Status", "Q3_City_Tier", "Q4_Financial_Dependents",
            "Q5_Employment_Status", "Q6_Monthly_Income", "Q7_Income_Stability", "Q8_Monthly_Expenditure",
            "Q9_Top_Spending_Categories", "Q10_Impulse_Purchase_Frequency", "Q11_Spending_Triggers",
            "Q12_Active_Subscriptions", "Q13_Statement_Review_Frequency", "Q14_Financial_Goals",
            "Q15_Savings_Percentage", "Q16_Financial_Challenges", "Q17_Financial_Confidence",
            "Q18_Financial_Stress_Level", "Q19_Budget_Behavior", "Q20_Finance_App_Usage",
            "Q21_Financial_Literacy", "Q22_Digital_Comfort_Level", "Q23_Preferred_Features",
            "Q24_Willingness_To_Pay"
        ]
        template_df = pd.DataFrame(columns=template_cols)
        template_csv = template_df.to_csv(index=False)
        st.download_button("📥 Download CSV Template", template_csv, "BudgetLife_Prediction_Template.csv", "text/csv")

        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.success(f"Uploaded {len(new_data)} records successfully!")
                st.dataframe(new_data.head(), use_container_width=True)

                if st.button("🔮 Run Predictions on All Records", use_container_width=True):
                    X_new = preprocess_new_customer(new_data, ordinal_maps, train_columns)
                    X_new_sc = scaler.transform(X_new)

                    predictions = rf_model.predict(X_new_sc)
                    probabilities = rf_model.predict_proba(X_new_sc)
                    pred_labels = le.inverse_transform(predictions)

                    results_df = new_data.copy()
                    results_df["Predicted_Category"] = pred_labels
                    for i, cls in enumerate(le.classes_):
                        results_df[f"Prob_{cls}"] = (probabilities[:, i] * 100).round(2)

                    # Assign marketing strategy
                    strategy_map = {
                        "Likely Adopter": "Direct Conversion — Premium trial + immediate onboarding",
                        "Persuadable": "Nurture & Educate — Free tier + content marketing",
                        "Unlikely": "Low Priority — Brand awareness + retarget later"
                    }
                    results_df["Recommended_Strategy"] = results_df["Predicted_Category"].map(strategy_map)

                    st.markdown("### Prediction Results")

                    # Summary
                    pred_summary = pd.Series(pred_labels).value_counts()
                    col1, col2, col3 = st.columns(3)
                    for i, cls in enumerate(le.classes_):
                        count = pred_summary.get(cls, 0)
                        [col1, col2, col3][i].metric(cls, f"{count} ({count/len(new_data)*100:.1f}%)")

                    fig = px.pie(values=pred_summary.values, names=pred_summary.index,
                               color=pred_summary.index,
                               color_discrete_map={"Likely Adopter": "#27AE60", "Persuadable": "#F9A825", "Unlikely": "#E74C3C"},
                               title="Prediction Distribution", hole=0.4)
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(results_df, use_container_width=True, height=400)

                    # Download results
                    csv_output = results_df.to_csv(index=False)
                    st.download_button("📥 Download Prediction Results", csv_output,
                                      "BudgetLife_Predictions.csv", "text/csv", use_container_width=True)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV matches the template format. Download the template above for reference.")
