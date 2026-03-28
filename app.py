import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score, r2_score, mean_squared_error, mean_absolute_error
)
from mlxtend.frequent_patterns import apriori, association_rules
import io

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="BudgetLife Analytics Dashboard", page_icon="💰", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    :root { --primary-color: #1B5E8C; --background-color: #FFFFFF; --secondary-background-color: #F7FBFE; --text-color: #0D3B5E; }
    .stApp { background-color: #FFFFFF !important; color: #0D3B5E !important; }

    /* Force all text dark globally — but NOT inside metric-cards */
    .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p { color: #0D3B5E !important; }
    .stApp div:not(.metric-card):not(.metric-value):not(.metric-label) { color: #0D3B5E; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #F7FBFE !important; }
    section[data-testid="stSidebar"] * { color: #0D3B5E !important; }
    section[data-testid="stSidebar"] hr { border-color: #C8DDE8 !important; }

    /* Metric boxes — force visible text */
    div[data-testid="stMetric"] { background: #F7FBFE !important; border: 1px solid #D0D8E0 !important; padding: 0.8rem; border-radius: 8px; }
    div[data-testid="stMetric"] * { color: #0D3B5E !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #1B5E8C !important; font-weight: 700 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: #0D3B5E !important; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #27AE60 !important; }

    /* Slider labels and values */
    .stSlider label, .stSlider div, .stSlider span { color: #0D3B5E !important; }
    .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color: #0D3B5E !important; }

    /* Selectbox, multiselect, radio */
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label { color: #0D3B5E !important; }
    .stSelectbox div[data-baseweb="select"] span { color: #0D3B5E !important; }

    /* Dataframe */
    .stDataFrame { color: #0D3B5E !important; }

    /* Buttons */
    .stButton > button { background-color: #1B5E8C; color: white !important; border: none; }
    .stButton > button:hover { background-color: #0D3B5E; color: white !important; }

    /* Expander */
    .streamlit-expanderHeader { color: #0D3B5E !important; }
    details summary span { color: #0D3B5E !important; }

    /* Tabs and captions */
    .stCaption, .stCaption p { color: #555555 !important; }

    /* Custom cards */
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0D3B5E !important; text-align: center; padding: 0.5rem 0; }
    .sub-header { font-size: 1rem; color: #1B5E8C !important; text-align: center; margin-bottom: 1.5rem; }
    .metric-card { background: linear-gradient(135deg, #1B5E8C 0%, #0D3B5E 100%); padding: 1.2rem; border-radius: 12px; color: #FFFFFF !important; text-align: center; box-shadow: 0 4px 15px rgba(13,59,94,0.2); }
    .stApp .metric-card, .stApp .metric-card div, .stApp .metric-card span, .stApp .metric-card p, .stApp .metric-card * { color: #FFFFFF !important; }
    .stApp .metric-value { font-size: 2rem; font-weight: 700; color: #FFFFFF !important; }
    .stApp .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 0.3rem; color: #D0E8F5 !important; }
    .insight-box { background: #F0F8FF; border-left: 4px solid #1B5E8C; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: 1rem 0; font-size: 0.95rem; color: #0D3B5E !important; }
    .insight-box * { color: #0D3B5E !important; }
    .strategy-box { background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%); border-left: 4px solid #27AE60; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: 0.8rem 0; color: #0D3B5E !important; }
    .strategy-box * { color: #0D3B5E !important; }
    .warning-box { background: #FFF8E1; border-left: 4px solid #F9A825; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: 0.8rem 0; color: #0D3B5E !important; }
    .warning-box * { color: #0D3B5E !important; }
</style>
""", unsafe_allow_html=True)

COLORS = ["#1B5E8C", "#27AE60", "#E67E22", "#8E44AD", "#E74C3C", "#16A085", "#2C3E50", "#F39C12"]

# ═══════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    return pd.read_csv("BudgetLife_Synthetic_Dataset.csv")

ORDINAL_MAPS = {
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
NOMINAL_COLS = ["Q2_Marital_Dependent_Status", "Q5_Employment_Status", "Q19_Budget_Behavior", "Q20_Finance_App_Usage", "Q22_Digital_Comfort_Level"]
MULTI_COLS = ["Q9_Top_Spending_Categories", "Q11_Spending_Triggers", "Q14_Financial_Goals", "Q16_Financial_Challenges", "Q23_Preferred_Features"]

@st.cache_data
def preprocess_for_ml(df):
    df_enc = df.drop(columns=["Respondent_ID", "Persona_Tag"], errors="ignore").copy()
    for col in MULTI_COLS:
        if col in df_enc.columns:
            unique_vals = set()
            df_enc[col].dropna().apply(lambda x: unique_vals.update(x.split("|")))
            for val in unique_vals:
                safe = f"{col}__{val.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('/', '_')}"
                df_enc[safe] = df_enc[col].apply(lambda x: 1 if val in str(x).split("|") else 0)
            df_enc.drop(columns=[col], inplace=True)
    for col, order in ORDINAL_MAPS.items():
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].map({v: i for i, v in enumerate(order)}).fillna(0).astype(int)
    for col in NOMINAL_COLS:
        if col in df_enc.columns:
            dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=True)
            df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)
    if "Q25_BudgetLife_Interest" in df_enc.columns:
        target_map = {"Definitely Yes": "Likely Adopter", "Probably Yes": "Likely Adopter", "Maybe": "Persuadable", "Probably No": "Unlikely", "Definitely No": "Unlikely"}
        df_enc["Target"] = df_enc["Q25_BudgetLife_Interest"].map(target_map)
        df_enc.drop(columns=["Q25_BudgetLife_Interest"], inplace=True)
    return df_enc

def preprocess_new_customer(new_df, train_columns):
    df_enc = new_df.copy()
    for col in MULTI_COLS:
        if col in df_enc.columns:
            unique_vals = set()
            df_enc[col].dropna().apply(lambda x: unique_vals.update(str(x).split("|")))
            for val in unique_vals:
                safe = f"{col}__{val.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('/', '_')}"
                df_enc[safe] = df_enc[col].apply(lambda x: 1 if val in str(x).split("|") else 0)
            df_enc.drop(columns=[col], inplace=True)
    for col, order in ORDINAL_MAPS.items():
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].map({v: i for i, v in enumerate(order)}).fillna(0).astype(int)
    for col in NOMINAL_COLS:
        if col in df_enc.columns:
            dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=True)
            df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)
    if "Q25_BudgetLife_Interest" in df_enc.columns:
        df_enc.drop(columns=["Q25_BudgetLife_Interest"], inplace=True)
    for c in train_columns:
        if c not in df_enc.columns:
            df_enc[c] = 0
    return df_enc[train_columns]

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file 'BudgetLife_Synthetic_Dataset.csv' not found.")
    st.stop()

df_ml = preprocess_for_ml(df)

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
        ["📊 Overview & Descriptive", "🔍 Exploratory Data Analysis", "🔬 Clustering Analysis",
         "🔗 Association Rule Mining", "🎯 Classification", "📈 Regression Analysis",
         "💡 Prescriptive Strategy", "🔮 New Customer Predictor"],
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

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Q1_Age_Group", color="Q1_Age_Group", color_discrete_sequence=COLORS,
                          title="Age Distribution", category_orders={"Q1_Age_Group": ["18–24", "25–34", "35–44", "45–54", "55+"]})
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 The 25–34 age group dominates our respondent pool, reflecting India's digitally active young workforce — BudgetLife's primary addressable market.")

    with col2:
        fig = px.pie(df, names="Q3_City_Tier", title="City Tier Distribution", color_discrete_sequence=COLORS, hole=0.4)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Metro and Tier 2 cities account for the majority, indicating our initial go-to-market should focus on urban India where digital finance adoption is highest.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Q6_Monthly_Income", color="Q6_Monthly_Income", color_discrete_sequence=COLORS,
                          title="Monthly Income Distribution",
                          category_orders={"Q6_Monthly_Income": ["Below ₹20,000", "₹20,001–₹50,000", "₹50,001–₹1,00,000", "₹1,00,001–₹2,00,000", "Above ₹2,00,000"]})
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 The ₹50K–₹1L bracket is the largest segment — these are middle-income earners with enough income to benefit from optimization but not enough to ignore overspending.")

    with col2:
        fig = px.histogram(df, x="Q15_Savings_Percentage", color="Q15_Savings_Percentage", color_discrete_sequence=COLORS,
                          title="Savings Rate Distribution",
                          category_orders={"Q15_Savings_Percentage": ["0% (No savings)", "1–10%", "11–20%", "21–30%", "Above 30%"]})
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Over 35% of respondents save 10% or less of their income — this is the pain point BudgetLife can directly address with savings recommendations.")

    st.markdown("### 🛒 Top Spending Categories")
    spending_cats = df["Q9_Top_Spending_Categories"].str.split("|").explode().value_counts()
    fig = px.bar(x=spending_cats.values, y=spending_cats.index, orientation="h", color=spending_cats.values,
                color_continuous_scale="Teal", title="Most Common Spending Categories Across All Respondents")
    fig.update_layout(height=420, yaxis_title="", xaxis_title="Frequency", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Groceries, Entertainment, and Rent dominate spending — BudgetLife's categorization engine must excel at distinguishing these three to deliver immediate value.")

    st.markdown("### 🧠 Financial Confidence vs. Stress Heatmap")
    cross = pd.crosstab(df["Q17_Financial_Confidence"], df["Q18_Financial_Stress_Level"])
    fig = px.imshow(cross, text_auto=True, color_continuous_scale="Blues", title="Financial Confidence × Financial Stress Cross-Tabulation")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 The highest concentration is at moderate confidence + moderate stress — these are 'aware but struggling' users who know they need help, making them the most receptive to BudgetLife.")

    st.markdown("### 🎯 BudgetLife Interest by Persona")
    interest_persona = pd.crosstab(df["Persona_Tag"], df["Q25_BudgetLife_Interest"], normalize="index") * 100
    interest_order = ["Definitely Yes", "Probably Yes", "Maybe", "Probably No", "Definitely No"]
    interest_persona = interest_persona.reindex(columns=[c for c in interest_order if c in interest_persona.columns])
    fig = px.bar(interest_persona, barmode="stack", color_discrete_sequence=["#27AE60", "#82E0AA", "#F9E79F", "#E67E22", "#E74C3C"],
                title="Adoption Interest Distribution by Persona (%)")
    fig.update_layout(height=420, yaxis_title="Percentage (%)", xaxis_title="", xaxis_tickangle=-20, legend_title="Interest Level")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 'Aspiring Digital-First Millennials' show the highest adoption intent while 'Retired/Senior Citizens' show the lowest — this confirms our primary vs. deprioritized segments.")


# ═══════════════════════════════════════════════════════════════
# TAB 2: EDA (NEW)
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "🔍 Exploratory Data Analysis":
    st.markdown('<div class="main-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive into data distributions, correlations, and patterns before modeling</div>', unsafe_allow_html=True)

    # Dataset overview
    st.markdown("### 📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Multi-Select Cols", "5")
    c4.metric("Missing Values", f"{df.isnull().sum().sum()}")

    st.caption("🔎 The dataset is complete with zero missing values — our synthetic generation ensured every respondent answered all 25 questions, so no imputation is needed.")

    with st.expander("📄 View Raw Data Sample (First 20 Rows)"):
        st.dataframe(df.head(20), use_container_width=True, height=350)

    st.markdown("---")

    # Distribution analysis
    st.markdown("### 📊 Single-Variable Distributions")
    st.caption("🔎 These distributions reveal the shape and spread of key survey responses — skewness here signals which customer profiles dominate our sample.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Q5_Employment_Status", color="Q5_Employment_Status", color_discrete_sequence=COLORS,
                          title="Employment Status Distribution")
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Private-sector salaried employees form the largest group, followed by self-employed — BudgetLife's UX should cater to both fixed-income and variable-income users.")

    with col2:
        fig = px.histogram(df, x="Q2_Marital_Dependent_Status", color="Q2_Marital_Dependent_Status",
                          color_discrete_sequence=COLORS, title="Marital & Dependent Status")
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 'Married with children' is the largest life-stage group — these users face the highest budgeting pressure, making them ideal candidates for BudgetLife's goal engine.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Q10_Impulse_Purchase_Frequency", color="Q10_Impulse_Purchase_Frequency",
                          color_discrete_sequence=COLORS, title="Impulse Purchase Frequency",
                          category_orders={"Q10_Impulse_Purchase_Frequency": ["Never", "Rarely (1–2 times/month)", "Sometimes (3–5 times/month)", "Often (6+ times/month)"]})
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Nearly 70% of respondents make impulse purchases at least sometimes — this validates the need for BudgetLife's predictive spending alerts.")

    with col2:
        fig = px.histogram(df, x="Q22_Digital_Comfort_Level", color="Q22_Digital_Comfort_Level",
                          color_discrete_sequence=COLORS, title="Digital Comfort Level (Linking Bank to Apps)")
        fig.update_layout(showlegend=False, height=380, xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 A near-even split between comfortable and uncomfortable users tells us that trust-building features (encryption badges, privacy controls) are not optional — they're essential.")

    st.markdown("---")

    # Bivariate analysis
    st.markdown("### 🔄 Bivariate Analysis — Income vs. Key Variables")
    st.caption("🔎 Cross-tabulating income against other variables reveals which financial behaviors are income-dependent and which are universal pain points.")

    col1, col2 = st.columns(2)
    with col1:
        cross1 = pd.crosstab(df["Q6_Monthly_Income"], df["Q15_Savings_Percentage"], normalize="index") * 100
        savings_order = ["0% (No savings)", "1–10%", "11–20%", "21–30%", "Above 30%"]
        cross1 = cross1.reindex(columns=[c for c in savings_order if c in cross1.columns])
        income_order = ["Below ₹20,000", "₹20,001–₹50,000", "₹50,001–₹1,00,000", "₹1,00,001–₹2,00,000", "Above ₹2,00,000"]
        cross1 = cross1.reindex([c for c in income_order if c in cross1.index])
        fig = px.bar(cross1, barmode="stack", color_discrete_sequence=["#E74C3C", "#E67E22", "#F9E79F", "#82E0AA", "#27AE60"],
                    title="Savings Rate by Income Bracket (%)")
        fig.update_layout(height=420, yaxis_title="%", xaxis_tickangle=-15, legend_title="Savings %")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Lower-income groups have the highest zero-savings rates — but even high-income groups show 15-20% saving only 0-10%, proving that overspending is not just an income problem.")

    with col2:
        cross2 = pd.crosstab(df["Q6_Monthly_Income"], df["Q25_BudgetLife_Interest"], normalize="index") * 100
        int_order = ["Definitely Yes", "Probably Yes", "Maybe", "Probably No", "Definitely No"]
        cross2 = cross2.reindex(columns=[c for c in int_order if c in cross2.columns])
        cross2 = cross2.reindex([c for c in income_order if c in cross2.index])
        fig = px.bar(cross2, barmode="stack", color_discrete_sequence=["#27AE60", "#82E0AA", "#F9E79F", "#E67E22", "#E74C3C"],
                    title="BudgetLife Interest by Income Bracket (%)")
        fig.update_layout(height=420, yaxis_title="%", xaxis_tickangle=-15, legend_title="Interest")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Mid-income brackets show the highest 'Definitely Yes' rates — they earn enough to value financial tools but face enough pressure to need them.")

    st.markdown("---")

    # Spending triggers analysis
    st.markdown("### 🧠 Spending Triggers Deep-Dive")
    triggers = df["Q11_Spending_Triggers"].str.split("|").explode().value_counts()
    fig = px.bar(x=triggers.values, y=triggers.index, orientation="h", color=triggers.values,
                color_continuous_scale="Reds", title="What Triggers Unplanned Spending?")
    fig.update_layout(height=400, yaxis_title="", xaxis_title="Frequency", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 'Flash sales & offers' and 'Convenience spending' top the list — BudgetLife should specifically flag impulse-prone categories like food delivery and e-commerce during high-sale periods.")

    # Financial goals
    st.markdown("### 🎯 Financial Goals Distribution")
    goals = df["Q14_Financial_Goals"].str.split("|").explode().value_counts()
    fig = px.bar(x=goals.values, y=goals.index, orientation="h", color=goals.values,
                color_continuous_scale="Greens", title="What Are People Saving For?")
    fig.update_layout(height=400, yaxis_title="", xaxis_title="Frequency", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 'Invest & Grow Wealth' and 'Build Emergency Fund' lead — BudgetLife's goal engine should prominently feature these two as default templates during onboarding.")

    st.markdown("---")

    # Correlation heatmap
    st.markdown("### 🔥 Correlation Heatmap (Ordinal Features)")
    st.caption("🔎 This heatmap shows how numerical survey features relate to each other — strong correlations guide which features to prioritize in our ML models.")
    corr_cols = list(ORDINAL_MAPS.keys())
    corr_df = df_ml[[c for c in corr_cols if c in df_ml.columns]].copy()
    corr_labels = ["Age", "City Tier", "Dependents", "Income", "Income Stability", "Expenditure",
                   "Impulse Freq", "Subscriptions", "Statement Review", "Savings %",
                   "Confidence", "Stress", "Literacy", "WTP"]
    corr_matrix = corr_df.corr()
    corr_matrix.index = corr_labels[:len(corr_matrix)]
    corr_matrix.columns = corr_labels[:len(corr_matrix)]
    fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                   title="Pairwise Correlation Between Ordinal Survey Features")
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Income and expenditure show the strongest positive correlation (expected). Financial literacy correlates with confidence — literate users feel more in control. Stress shows weak negative correlation with savings, confirming that financial anxiety is tied to poor saving habits.")

    # Persona distribution
    st.markdown("### 👥 Persona Distribution")
    persona_counts = df["Persona_Tag"].value_counts()
    fig = px.bar(x=persona_counts.index, y=persona_counts.values, color=persona_counts.index,
                color_discrete_sequence=COLORS, title="Respondent Count by Persona Archetype")
    fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-20, xaxis_title="", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 'Young Urban Professionals' form the largest group (22%), followed by 'Middle-Income Family Providers' (18%) — these two personas alone represent 40% of our addressable market.")

    st.markdown("---")

    # ── DRILL-DOWN CHARTS ──
    st.markdown("### 🔽 Drill-Down Analysis — Interactive Exploration")
    st.caption("🔎 Click on any segment in the sunburst chart to drill down into sub-categories. Click the center to zoom back out. This reveals how demographics, income, and behavior nest within each other.")

    # Sunburst: City Tier → Income → Savings → BudgetLife Interest
    st.markdown("#### Sunburst: City Tier → Income → Savings → BudgetLife Interest")
    drill_df = df[["Q3_City_Tier", "Q6_Monthly_Income", "Q15_Savings_Percentage", "Q25_BudgetLife_Interest"]].copy()
    drill_df.columns = ["City Tier", "Income", "Savings", "BudgetLife Interest"]
    fig = px.sunburst(
        drill_df, path=["City Tier", "Income", "Savings", "BudgetLife Interest"],
        color="City Tier",
        color_discrete_map={"Metro (Tier 1)": "#1B5E8C", "Tier 2": "#27AE60", "Tier 3": "#E67E22", "Rural": "#8E44AD"},
        title="Drill-Down: City Tier → Income Bracket → Savings Rate → Adoption Interest"
    )
    fig.update_layout(height=600)
    fig.update_traces(textinfo="label+percent parent", insidetextorientation="radial")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Click any ring segment to drill into its breakdown. For example, clicking 'Metro (Tier 1)' reveals how metro users split across income brackets, then how each income bracket splits by savings rate, and finally how each savings group feels about BudgetLife. This multi-level view shows that metro + high-income + low-savings users are the highest-opportunity segment.")

    # Sunburst: Persona → Budget Behavior → Stress → Interest
    st.markdown("#### Sunburst: Persona → Budget Behavior → Financial Stress → Adoption")
    drill_df2 = df[["Persona_Tag", "Q19_Budget_Behavior", "Q18_Financial_Stress_Level", "Q25_BudgetLife_Interest"]].copy()
    drill_df2.columns = ["Persona", "Budget Behavior", "Stress Level", "BudgetLife Interest"]
    fig2 = px.sunburst(
        drill_df2, path=["Persona", "Budget Behavior", "Stress Level", "BudgetLife Interest"],
        color="Persona", color_discrete_sequence=COLORS,
        title="Drill-Down: Persona → Budget Behavior → Stress Level → Adoption Interest"
    )
    fig2.update_layout(height=600)
    fig2.update_traces(textinfo="label+percent parent", insidetextorientation="radial")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔎 This drill-down reveals the behavioral pipeline: which personas lack budgets, which of those are stressed, and which stressed non-budgeters are likely adopters. The path Persona → 'No, but I want to' → High Stress → 'Definitely Yes' identifies BudgetLife's highest-conversion micro-segment.")

    # Treemap: Employment → Income → Impulse → Subscriptions
    st.markdown("#### Treemap: Employment → Income → Impulse Spending → Subscriptions")
    drill_df3 = df[["Q5_Employment_Status", "Q6_Monthly_Income", "Q10_Impulse_Purchase_Frequency", "Q12_Active_Subscriptions"]].copy()
    drill_df3.columns = ["Employment", "Income", "Impulse Spending", "Subscriptions"]
    drill_df3_agg = drill_df3.groupby(["Employment", "Income", "Impulse Spending", "Subscriptions"]).size().reset_index(name="Count")
    fig3 = px.treemap(
        drill_df3_agg, path=[px.Constant("All Respondents"), "Employment", "Income", "Impulse Spending", "Subscriptions"],
        values="Count", color="Count", color_continuous_scale="Teal",
        title="Drill-Down Treemap: Employment → Income → Impulse Frequency → Subscription Count"
    )
    fig3.update_layout(height=600)
    fig3.update_traces(textinfo="label+value")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("🔎 Click any box to drill into its children. This treemap reveals spending behavior patterns within each employment type. Salaried private employees with ₹50K–₹1L income who impulse-spend 'Sometimes' and have 3–5 subscriptions form the single largest actionable sub-group for BudgetLife's subscription detection feature.")


# ═══════════════════════════════════════════════════════════════
# TAB 3: CLUSTERING
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "🔬 Clustering Analysis":
    st.markdown('<div class="main-header">🔬 Diagnostic Analysis — Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discovering natural customer segments using K-Means Clustering</div>', unsafe_allow_html=True)

    cluster_features = [c for c in ["Q1_Age_Group", "Q3_City_Tier", "Q4_Financial_Dependents", "Q6_Monthly_Income",
        "Q7_Income_Stability", "Q8_Monthly_Expenditure", "Q10_Impulse_Purchase_Frequency",
        "Q12_Active_Subscriptions", "Q13_Statement_Review_Frequency", "Q15_Savings_Percentage",
        "Q17_Financial_Confidence", "Q18_Financial_Stress_Level", "Q21_Financial_Literacy",
        "Q24_Willingness_To_Pay"] if c in df_ml.columns]

    scaler_clust = StandardScaler()
    X_scaled = scaler_clust.fit_transform(df_ml[cluster_features])

    st.markdown("### Optimal Cluster Selection")
    st.caption("🔎 We use two methods together: the Elbow method finds where adding more clusters stops reducing error, and the Silhouette score measures how well-separated the clusters are.")

    col1, col2 = st.columns(2)
    inertias, sil_scores = [], []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    with col1:
        fig = go.Figure(go.Scatter(x=list(K_range), y=inertias, mode="lines+markers", marker=dict(size=10, color="#1B5E8C"), line=dict(width=3, color="#1B5E8C")))
        fig.update_layout(title="Elbow Method — Inertia vs. K", xaxis_title="Number of Clusters (K)", yaxis_title="Inertia", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 The 'elbow' appears where the curve bends — beyond this point, adding more clusters gives diminishing returns in reducing within-cluster variance.")

    with col2:
        fig = go.Figure(go.Scatter(x=list(K_range), y=sil_scores, mode="lines+markers", marker=dict(size=10, color="#27AE60"), line=dict(width=3, color="#27AE60")))
        fig.update_layout(title="Silhouette Score vs. K", xaxis_title="Number of Clusters (K)", yaxis_title="Silhouette Score", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔎 Higher silhouette score = better-defined clusters. The peak tells us the most natural number of customer segments in our data.")

    optimal_k = list(K_range)[np.argmax(sil_scores)]
    st.markdown(f'<div class="insight-box">📌 <b>Optimal K = {optimal_k}</b> — Highest silhouette score of {max(sil_scores):.4f}.</div>', unsafe_allow_html=True)

    n_clusters = st.slider("Select number of clusters", 2, 8, optimal_k)
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = km_final.fit_predict(X_scaled)

    st.markdown("### Cluster Size Distribution")
    cluster_counts = df["Cluster"].value_counts().sort_index()
    fig = px.bar(x=[f"Cluster {i}" for i in cluster_counts.index], y=cluster_counts.values,
                color=[f"Cluster {i}" for i in cluster_counts.index], color_discrete_sequence=COLORS, title="Respondents per Cluster")
    fig.update_layout(showlegend=False, height=350, xaxis_title="", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Relatively balanced cluster sizes indicate the algorithm found genuine segments rather than forcing unnatural groupings.")

    st.markdown("### Cluster Profile Comparison (Radar Chart)")
    st.caption("🔎 Each polygon represents a cluster's average profile — wider shapes mean higher values. Where polygons diverge reveals what differentiates each segment.")
    radar_features = [c for c in ["Q6_Monthly_Income", "Q8_Monthly_Expenditure", "Q15_Savings_Percentage",
                      "Q10_Impulse_Purchase_Frequency", "Q17_Financial_Confidence",
                      "Q18_Financial_Stress_Level", "Q21_Financial_Literacy", "Q24_Willingness_To_Pay"] if c in df_ml.columns]
    radar_labels = ["Income", "Expenditure", "Savings", "Impulse Spending", "Confidence", "Stress", "Literacy", "WTP"]

    radar_df = df_ml.copy()
    radar_df["Cluster"] = df["Cluster"]
    cluster_means = radar_df.groupby("Cluster")[radar_features].mean()
    mms = MinMaxScaler()
    cluster_means_norm = pd.DataFrame(mms.fit_transform(cluster_means), columns=radar_features, index=cluster_means.index)

    fig = go.Figure()
    for i in range(n_clusters):
        vals = cluster_means_norm.loc[i].tolist() + [cluster_means_norm.loc[i].tolist()[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=radar_labels + [radar_labels[0]], fill="toself", name=f"Cluster {i}", line=dict(color=COLORS[i % len(COLORS)])))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Cluster Profiles — Normalized", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cluster × BudgetLife Adoption Interest")
    cluster_interest = pd.crosstab(df["Cluster"], df["Q25_BudgetLife_Interest"], normalize="index") * 100
    int_order = ["Definitely Yes", "Probably Yes", "Maybe", "Probably No", "Definitely No"]
    cluster_interest = cluster_interest.reindex(columns=[c for c in int_order if c in cluster_interest.columns])
    cluster_interest.index = [f"Cluster {i}" for i in cluster_interest.index]
    fig = px.bar(cluster_interest, barmode="stack", color_discrete_sequence=["#27AE60", "#82E0AA", "#F9E79F", "#E67E22", "#E74C3C"], title="Adoption Likelihood by Cluster")
    fig.update_layout(height=400, yaxis_title="%", xaxis_title="", legend_title="Interest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Clusters with the highest green (Definitely/Probably Yes) are our priority targets — these are the segments where marketing spend will have the highest ROI.")

    st.markdown("### Cluster Summary Statistics")
    summary_cols = {"Q6_Monthly_Income": "Avg Income", "Q8_Monthly_Expenditure": "Avg Expenditure", "Q15_Savings_Percentage": "Avg Savings",
                    "Q18_Financial_Stress_Level": "Avg Stress", "Q21_Financial_Literacy": "Avg Literacy", "Q24_Willingness_To_Pay": "Avg WTP"}
    summary = radar_df.groupby("Cluster")[list(summary_cols.keys())].mean().round(2)
    summary.columns = list(summary_cols.values())
    summary["Size"] = cluster_counts.values
    summary.index = [f"Cluster {i}" for i in summary.index]
    st.dataframe(summary, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4: ASSOCIATION RULE MINING
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "🔗 Association Rule Mining":
    st.markdown('<div class="main-header">🔗 Diagnostic Analysis — Association Rule Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discovering hidden relationships between spending patterns, challenges, and feature preferences</div>', unsafe_allow_html=True)

    multi_col_map = {"Q9_Top_Spending_Categories": "Spend", "Q11_Spending_Triggers": "Trigger", "Q14_Financial_Goals": "Goal",
                     "Q16_Financial_Challenges": "Challenge", "Q23_Preferred_Features": "Feature"}
    transactions = pd.DataFrame()
    for col, prefix in multi_col_map.items():
        items = df[col].str.get_dummies(sep="|")
        items.columns = [f"{prefix}: {c}" for c in items.columns]
        transactions = pd.concat([transactions, items], axis=1)

    transactions["High Impulse Spender"] = df["Q10_Impulse_Purchase_Frequency"].isin(["Sometimes (3–5 times/month)", "Often (6+ times/month)"]).astype(int)
    transactions["Low Saver (0-10%)"] = df["Q15_Savings_Percentage"].isin(["0% (No savings)", "1–10%"]).astype(int)
    transactions["No Budget"] = df["Q19_Budget_Behavior"].isin(["No, but I want to", "No, and I don't plan to"]).astype(int)
    transactions["Never Used Finance App"] = (df["Q20_Finance_App_Usage"] == "No, never used one").astype(int)
    transactions["High Financial Stress"] = df["Q18_Financial_Stress_Level"].isin(["4 – Often", "5 – Almost always"]).astype(int)

    min_support = st.slider("Minimum Support Threshold", 0.03, 0.20, 0.07, 0.01)
    min_confidence = st.slider("Minimum Confidence Threshold", 0.30, 0.80, 0.40, 0.05)
    st.caption("🔎 **Support** = how frequently the items appear together. **Confidence** = given the antecedent, how often the consequent follows. Adjust sliders to explore more or fewer rules.")

    frequent_items = apriori(transactions, min_support=min_support, use_colnames=True)

    if len(frequent_items) == 0:
        st.warning("No frequent itemsets found. Try lowering the support threshold.")
    else:
        rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_items))
        rules = rules[rules["lift"] > 1.0]
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

        if len(rules) == 0:
            st.warning("No rules found above thresholds. Try adjusting the sliders.")
        else:
            st.markdown(f"### Discovered {len(rules)} Association Rules")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rules", len(rules))
            c2.metric("Avg Confidence", f"{rules['confidence'].mean():.2f}")
            c3.metric("Avg Lift", f"{rules['lift'].mean():.2f}")

            st.markdown("### Top 20 Rules by Lift")
            st.caption("🔎 **Lift > 1** means the items appear together more often than by chance. Higher lift = stronger and more actionable relationship.")
            top_rules = rules.nlargest(20, "lift")[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)
            top_rules.columns = ["If (Antecedent)", "Then (Consequent)", "Support", "Confidence", "Lift"]
            top_rules = top_rules.round(3)
            st.dataframe(top_rules, use_container_width=True, height=500)

            st.markdown("### Confidence vs. Lift Scatter Plot")
            fig = px.scatter(rules, x="confidence", y="lift", size="support", color="lift",
                           color_continuous_scale="Viridis", hover_data=["antecedents", "consequents"],
                           title="Each dot is a rule — top-right corner = strongest actionable rules")
            fig.update_layout(height=480, xaxis_title="Confidence", yaxis_title="Lift")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🔎 Rules in the upper-right quadrant have both high confidence and high lift — these are the patterns BudgetLife should act on for personalized recommendations.")

            st.markdown("### Support vs. Confidence Scatter Plot")
            fig = px.scatter(rules, x="support", y="confidence", size="lift", color="confidence",
                           color_continuous_scale="Teal", hover_data=["antecedents", "consequents"],
                           title="Balancing rule frequency (support) with predictive strength (confidence)")
            fig.update_layout(height=480, xaxis_title="Support", yaxis_title="Confidence")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🔎 High-support rules affect many users (broad campaigns). High-confidence rules are more targeted (personalized nudges). The ideal strategy uses both.")

            st.markdown("### Top 15 Rules by Confidence")
            top_conf = rules.nlargest(15, "confidence")
            fig = px.bar(top_conf, x="confidence", y=top_conf["antecedents"].str[:45] + " → " + top_conf["consequents"].str[:45],
                        orientation="h", color="lift", color_continuous_scale="Oranges", title="Highest Confidence Rules")
            fig.update_layout(height=550, yaxis_title="", xaxis_title="Confidence")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🔎 These are our most reliable predictions — when users exhibit the 'If' pattern, they almost certainly exhibit the 'Then' behavior too.")


# ═══════════════════════════════════════════════════════════════
# TAB 5: CLASSIFICATION
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
    scaler_cls = StandardScaler()
    X_train_sc = scaler_cls.fit_transform(X_train)
    X_test_sc = scaler_cls.transform(X_test)

    # FIX: Removed multi_class parameter (deprecated in newer scikit-learn)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    rf.fit(X_train_sc, y_train)
    lr.fit(X_train_sc, y_train)
    rf_pred = rf.predict(X_test_sc)
    lr_pred = lr.predict(X_test_sc)

    st.markdown("### Model Performance Comparison")
    st.caption("🔎 We train two models and compare — Random Forest captures complex non-linear patterns while Logistic Regression provides interpretable coefficients.")

    col1, col2 = st.columns(2)
    for col, name, y_pred in [(col1, "Random Forest", rf_pred), (col2, "Logistic Regression", lr_pred)]:
        with col:
            st.markdown(f"#### {name}")
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

    st.caption("🔎 **Accuracy** = overall correctness. **Precision** = of those predicted positive, how many were correct. **Recall** = of actual positives, how many were caught. **F1** = harmonic mean balancing precision & recall.")

    st.markdown("### Detailed Classification Report")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        st.dataframe(pd.DataFrame(classification_report(y_test, rf_pred, target_names=class_names, output_dict=True)).T.round(4), use_container_width=True)
    with col2:
        st.markdown("**Logistic Regression**")
        st.dataframe(pd.DataFrame(classification_report(y_test, lr_pred, target_names=class_names, output_dict=True)).T.round(4), use_container_width=True)

    st.markdown("### Confusion Matrices")
    st.caption("🔎 The diagonal shows correct predictions. Off-diagonal cells show misclassifications — larger off-diagonal numbers reveal which classes the model confuses.")
    col1, col2 = st.columns(2)
    for col_obj, name, preds in [(col1, "Random Forest", rf_pred), (col2, "Logistic Regression", lr_pred)]:
        with col_obj:
            cm = confusion_matrix(y_test, preds)
            fig = px.imshow(cm, text_auto=True, x=class_names, y=class_names, color_continuous_scale="Blues", title=f"{name}")
            fig.update_layout(height=380, xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ROC Curves (One-vs-Rest)")
    st.caption("🔎 ROC curves show the trade-off between true positive rate and false positive rate. AUC closer to 1.0 = better discrimination between classes.")
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Random Forest ROC", "Logistic Regression ROC"])
    for idx, (model, model_name) in enumerate([(rf, "Random Forest"), (lr, "Logistic Regression")]):
        y_prob = model.predict_proba(X_test_sc)
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{cls} (AUC={roc_auc:.3f})", line=dict(color=COLORS[i % len(COLORS)]), showlegend=(idx == 0)), row=1, col=idx + 1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), showlegend=False), row=1, col=idx + 1)
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Importance — Random Forest (Top 20)")
    st.caption("🔎 These are the features that most influence the model's predictions — the higher the bar, the more that question drives whether someone will adopt BudgetLife.")
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(20)
    fig = px.bar(x=feat_imp.values, y=feat_imp.index, orientation="h", color=feat_imp.values, color_continuous_scale="Teal", title="Top 20 Features Driving Adoption Prediction")
    fig.update_layout(height=550, yaxis_title="", xaxis_title="Importance", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.session_state["rf_model"] = rf
    st.session_state["scaler"] = scaler_cls
    st.session_state["le"] = le
    st.session_state["train_columns"] = list(X.columns)


# ═══════════════════════════════════════════════════════════════
# TAB 6: REGRESSION
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "📈 Regression Analysis":
    st.markdown('<div class="main-header">📈 Predictive Analysis — Regression Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicting monthly expenditure, savings potential, and willingness to pay</div>', unsafe_allow_html=True)

    regression_targets = {"Monthly Expenditure": "Q8_Monthly_Expenditure", "Savings Potential": "Q15_Savings_Percentage", "Willingness to Pay": "Q24_Willingness_To_Pay"}
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
        results[target_name] = {"r2": r2_score(y_te, y_pred), "rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
                                 "mae": mean_absolute_error(y_te, y_pred), "y_te": y_te, "y_pred": y_pred,
                                 "coefs": pd.Series(ridge.coef_, index=reg_features)}

    st.markdown("### Model Performance Summary")
    st.caption("🔎 **R²** = proportion of variance explained (1.0 = perfect). **RMSE** = average prediction error in the same units as the target. **MAE** = average absolute error.")
    perf_data = [{"Target": n, "R² Score": round(r["r2"], 4), "RMSE": round(r["rmse"], 4), "MAE": round(r["mae"], 4)} for n, r in results.items()]
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.markdown("### Actual vs. Predicted Plots")
    st.caption("🔎 Points close to the red diagonal line = accurate predictions. Scatter away from the line shows where the model struggles.")
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            fig = px.scatter(x=res["y_te"], y=res["y_pred"], title=f"{name}", labels={"x": "Actual", "y": "Predicted"}, color_discrete_sequence=[COLORS[i]])
            mx = max(max(res["y_te"]), max(res["y_pred"]))
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="red"), showlegend=False))
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Residual Analysis")
    st.caption("🔎 Residuals should be centered around zero and roughly bell-shaped. A skewed distribution suggests systematic prediction bias.")
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            residuals = res["y_te"].values - res["y_pred"]
            fig = px.histogram(x=residuals, nbins=30, title=f"{name} — Residuals", color_discrete_sequence=[COLORS[i]])
            fig.update_layout(height=350, xaxis_title="Residual", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Feature Coefficients (Ridge Regression)")
    st.caption("🔎 Positive coefficients push the prediction higher; negative coefficients push it lower. These reveal what drives each financial behavior.")
    target_select = st.selectbox("Select Target Variable", list(results.keys()))
    coefs = results[target_select]["coefs"]
    top_coefs = pd.concat([coefs.nlargest(10), coefs.nsmallest(10)]).sort_values()
    fig = px.bar(x=top_coefs.values, y=top_coefs.index, orientation="h", color=top_coefs.values, color_continuous_scale="RdBu_r",
                title=f"Top Positive & Negative Predictors — {target_select}")
    fig.update_layout(height=550, yaxis_title="", xaxis_title="Coefficient", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 7: PRESCRIPTIVE STRATEGY
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "💡 Prescriptive Strategy":
    st.markdown('<div class="main-header">💡 Prescriptive Analysis — Strategic Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Translating data insights into actionable business strategy</div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Target Segment Prioritization")
    st.caption("🔎 Revenue Score = Adoption Likelihood × Willingness to Pay — this single metric ranks which segments will generate the most revenue per marketing dollar spent.")

    persona_analysis = []
    for persona in df["Persona_Tag"].unique():
        p_df = df[df["Persona_Tag"] == persona]
        likely_pct = p_df["Q25_BudgetLife_Interest"].isin(["Definitely Yes", "Probably Yes"]).mean() * 100
        wtp_map = {"₹0 (Free only)": 0, "₹49–₹99": 74, "₹100–₹199": 150, "₹200–₹499": 350, "₹500+": 500}
        avg_wtp = p_df["Q24_Willingness_To_Pay"].map(wtp_map).mean()
        stress_map = {"1 – Never": 1, "2 – Rarely": 2, "3 – Sometimes": 3, "4 – Often": 4, "5 – Almost always": 5}
        avg_stress = p_df["Q18_Financial_Stress_Level"].map(stress_map).mean()
        digital_pct = p_df["Q22_Digital_Comfort_Level"].isin(["Very comfortable", "Somewhat comfortable"]).mean() * 100
        persona_analysis.append({"Segment": persona, "Size": len(p_df), "Adoption Likelihood (%)": round(likely_pct, 1),
            "Avg WTP (₹)": round(avg_wtp), "Avg Stress": round(avg_stress, 2), "Digital Comfort (%)": round(digital_pct, 1),
            "Revenue Score": round(likely_pct * avg_wtp / 100, 1)})

    pa_df = pd.DataFrame(persona_analysis).sort_values("Revenue Score", ascending=False)
    st.dataframe(pa_df, use_container_width=True, hide_index=True)

    fig = px.bar(pa_df, x="Segment", y="Revenue Score", color="Adoption Likelihood (%)", color_continuous_scale="Greens", title="Segment Revenue Score")
    fig.update_layout(height=420, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Segments with the tallest bars should receive the highest marketing budget allocation — they combine willingness to pay with genuine interest.")

    st.markdown("### Adoption vs. Willingness to Pay — Strategic Quadrant")
    fig = px.scatter(pa_df, x="Adoption Likelihood (%)", y="Avg WTP (₹)", size="Size", color="Segment", color_discrete_sequence=COLORS, text="Segment",
                    title="Where Each Segment Falls in the Strategy Matrix")
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=480)
    fig.add_hline(y=pa_df["Avg WTP (₹)"].median(), line_dash="dash", line_color="gray", annotation_text="Median WTP")
    fig.add_vline(x=pa_df["Adoption Likelihood (%)"].median(), line_dash="dash", line_color="gray", annotation_text="Median Adoption")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="strategy-box"><b>🟢 Top-Right (High Adoption + High WTP):</b> Primary target — invest heavily in acquisition. These users will pay and they want the product.</div>
    <div class="insight-box"><b>🔵 Top-Left (Low Adoption + High WTP):</b> Trust-building needed. Focus on credibility, security messaging, and referrals.</div>
    <div class="warning-box"><b>🟡 Bottom-Right (High Adoption + Low WTP):</b> Freemium funnel — they want it but won't pay yet. Convert through free tier, upsell later.</div>
    """, unsafe_allow_html=True)

    st.markdown("### 📦 Feature Demand Analysis")
    features_exploded = df["Q23_Preferred_Features"].str.split("|").explode()
    feat_demand = features_exploded.value_counts()
    fig = px.bar(x=feat_demand.values, y=feat_demand.index, orientation="h", color=feat_demand.values, color_continuous_scale="Teal", title="Feature Demand — What Customers Want Most")
    fig.update_layout(height=350, yaxis_title="", xaxis_title="Selection Count", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔎 Build and market the top 2 features first — they represent the highest demand and will drive initial adoption.")

    st.markdown("### 💰 Pricing Strategy")
    wtp_dist = df["Q24_Willingness_To_Pay"].value_counts()
    wtp_order = ["₹0 (Free only)", "₹49–₹99", "₹100–₹199", "₹200–₹499", "₹500+"]
    wtp_dist = wtp_dist.reindex(wtp_order)
    fig = px.bar(x=wtp_order, y=wtp_dist.values, color=wtp_dist.values, color_continuous_scale="Oranges", title="Willingness to Pay Distribution")
    fig.update_layout(height=380, xaxis_title="Price Tier", yaxis_title="Respondents", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    free_pct = (df["Q24_Willingness_To_Pay"] == "₹0 (Free only)").mean() * 100
    mid_pct = df["Q24_Willingness_To_Pay"].isin(["₹49–₹99", "₹100–₹199"]).mean() * 100
    st.markdown(f"""
    <div class="strategy-box">
    <b>Recommended Pricing — Freemium + Premium:</b><br>
    • <b>Free:</b> Basic tracking — captures {free_pct:.0f}% who won't pay initially<br>
    • <b>Plus (₹99/mo):</b> Subscription detection + alerts — targets the {mid_pct:.0f}% willing to pay ₹49–₹199<br>
    • <b>Pro (₹249/mo):</b> Full AI engine, health score, peer benchmarking — for high-value users
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 8: NEW CUSTOMER PREDICTOR
# ═══════════════════════════════════════════════════════════════

elif tab_choice == "🔮 New Customer Predictor":
    st.markdown('<div class="main-header">🔮 New Customer Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict adoption likelihood for new potential customers</div>', unsafe_allow_html=True)

    if "rf_model" not in st.session_state:
        st.markdown('<div class="warning-box">⚠️ <b>Models not trained yet.</b> Please visit the <b>🎯 Classification</b> tab first to train models, then return here.</div>', unsafe_allow_html=True)
        st.stop()

    rf_model = st.session_state["rf_model"]
    scaler = st.session_state["scaler"]
    le = st.session_state["le"]
    train_columns = st.session_state["train_columns"]

    pred_mode = st.radio("Prediction Mode", ["📝 Single Customer Entry", "📁 Bulk CSV Upload"], horizontal=True)

    if pred_mode == "📝 Single Customer Entry":
        st.markdown("### Enter Customer Profile")
        with st.form("single_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                q1 = st.selectbox("Age Group", ["18–24", "25–34", "35–44", "45–54", "55+"])
                q2 = st.selectbox("Marital & Dependent Status", ["Single, no dependents", "Single, with dependents", "Married, no children", "Married, with children", "Joint family (supporting elders)"])
                q3 = st.selectbox("City Tier", ["Metro (Tier 1)", "Tier 2", "Tier 3", "Rural"])
                q4 = st.selectbox("Financial Dependents", ["0", "1", "2", "3", "4+"])
                q5 = st.selectbox("Employment Status", ["Salaried (Private)", "Salaried (Government)", "Self-employed", "Freelancer or Gig Worker", "Student", "Unemployed", "Retired"])
                q6 = st.selectbox("Monthly Income", ["Below ₹20,000", "₹20,001–₹50,000", "₹50,001–₹1,00,000", "₹1,00,001–₹2,00,000", "Above ₹2,00,000"])
                q7 = st.selectbox("Income Stability", ["Very stable (fixed salary)", "Mostly stable with some variation", "Irregular but manageable", "Highly unpredictable"])
                q8 = st.selectbox("Monthly Expenditure", ["Below ₹10,000", "₹10,001–₹25,000", "₹25,001–₹50,000", "₹50,001–₹1,00,000", "Above ₹1,00,000"])
            with c2:
                q9 = st.multiselect("Top Spending (up to 3)", ["Rent & Housing", "Groceries & Food", "Transportation", "Healthcare", "Education", "Entertainment & Dining Out", "Subscriptions (OTT, Gym, etc.)", "Shopping & Fashion", "EMIs & Loan Repayments", "Other"], max_selections=3, default=["Groceries & Food"])
                q10 = st.selectbox("Impulse Purchase Frequency", ["Never", "Rarely (1–2 times/month)", "Sometimes (3–5 times/month)", "Often (6+ times/month)"])
                q11 = st.multiselect("Spending Triggers (up to 2)", ["Emotional stress or boredom", "Social pressure or peer influence", "Flash sales, discounts & offers", "Celebrations or festivals", "Convenience (food delivery, cabs, etc.)", "Social media influence", "I rarely spend impulsively"], max_selections=2, default=["Flash sales, discounts & offers"])
                q12 = st.selectbox("Active Subscriptions", ["None", "1–2", "3–5", "6+"])
                q13 = st.selectbox("Statement Review", ["Weekly", "Monthly", "Quarterly", "Rarely", "Never"])
                q14 = st.multiselect("Financial Goals (up to 3)", ["Build Emergency Fund", "Save for Vacation", "Pay Off Debt", "Invest & Grow Wealth", "Save for Major Purchase (Home, Car)", "Children's Education", "Retirement Planning", "No Specific Goal"], max_selections=3, default=["Build Emergency Fund"])
            with c3:
                q15 = st.selectbox("Savings %", ["0% (No savings)", "1–10%", "11–20%", "21–30%", "Above 30%"])
                q16 = st.multiselect("Challenges (up to 2)", ["Overspending", "Low or Irregular Income", "High EMIs or Debt", "Lack of Financial Knowledge", "No Budgeting Discipline", "Unexpected Expenses", "High Cost of Living"], max_selections=2, default=["Overspending"])
                q17 = st.selectbox("Financial Confidence", ["1 – Not confident at all", "2 – Slightly confident", "3 – Moderately confident", "4 – Very confident", "5 – Extremely confident"])
                q18 = st.selectbox("Financial Stress", ["1 – Never", "2 – Rarely", "3 – Sometimes", "4 – Often", "5 – Almost always"])
                q19 = st.selectbox("Budget Behavior", ["Yes, strictly", "Yes, loosely", "No, but I want to", "No, and I don't plan to"])
                q20 = st.selectbox("Finance App Usage", ["Yes, currently using one", "Used before but stopped", "No, never used one"])
                q21 = st.selectbox("Financial Literacy", ["1 – Very Low", "2 – Low", "3 – Average", "4 – High", "5 – Very High"])
                q22 = st.selectbox("Digital Comfort", ["Very comfortable", "Somewhat comfortable", "Neutral", "Somewhat uncomfortable", "Very uncomfortable"])
                q23 = st.multiselect("Preferred Features (up to 3)", ["Automatic Expense Tracking & Smart Categorization", "Subscription & Money Leak Detection", "Predictive Budget Alerts & Spending Forecasts", "Financial Health Score with Peer Benchmarking", "AI Savings Recommendations & Goal Engine"], max_selections=3, default=["Automatic Expense Tracking & Smart Categorization"])
                q24 = st.selectbox("Willingness to Pay", ["₹0 (Free only)", "₹49–₹99", "₹100–₹199", "₹200–₹499", "₹500+"])
            submitted = st.form_submit_button("🔮 Predict Adoption Likelihood", use_container_width=True)

        if submitted:
            new_row = {"Q1_Age_Group": q1, "Q2_Marital_Dependent_Status": q2, "Q3_City_Tier": q3, "Q4_Financial_Dependents": q4,
                "Q5_Employment_Status": q5, "Q6_Monthly_Income": q6, "Q7_Income_Stability": q7, "Q8_Monthly_Expenditure": q8,
                "Q9_Top_Spending_Categories": "|".join(q9), "Q10_Impulse_Purchase_Frequency": q10, "Q11_Spending_Triggers": "|".join(q11),
                "Q12_Active_Subscriptions": q12, "Q13_Statement_Review_Frequency": q13, "Q14_Financial_Goals": "|".join(q14),
                "Q15_Savings_Percentage": q15, "Q16_Financial_Challenges": "|".join(q16), "Q17_Financial_Confidence": q17,
                "Q18_Financial_Stress_Level": q18, "Q19_Budget_Behavior": q19, "Q20_Finance_App_Usage": q20,
                "Q21_Financial_Literacy": q21, "Q22_Digital_Comfort_Level": q22, "Q23_Preferred_Features": "|".join(q23), "Q24_Willingness_To_Pay": q24}

            X_new = preprocess_new_customer(pd.DataFrame([new_row]), train_columns)
            X_new_sc = scaler.transform(X_new)
            pred = rf_model.predict(X_new_sc)[0]
            proba = rf_model.predict_proba(X_new_sc)[0]
            pred_label = le.inverse_transform([pred])[0]

            st.markdown("---")
            color_map = {"Likely Adopter": "#27AE60", "Persuadable": "#F9A825", "Unlikely": "#E74C3C"}
            emoji_map = {"Likely Adopter": "✅", "Persuadable": "🟡", "Unlikely": "❌"}
            st.markdown(f'<div style="background:{color_map.get(pred_label,"#1B5E8C")}22;border-left:6px solid {color_map.get(pred_label,"#1B5E8C")};padding:1.5rem;border-radius:0 12px 12px 0;margin:1rem 0;"><h2 style="margin:0;color:{color_map.get(pred_label,"#1B5E8C")}">{emoji_map.get(pred_label,"")} {pred_label}</h2></div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            for i, cls in enumerate(le.classes_):
                [c1, c2, c3][i].metric(cls, f"{proba[i]*100:.1f}%")

            fig = px.bar(x=le.classes_, y=proba * 100, color=le.classes_,
                        color_discrete_map={"Likely Adopter": "#27AE60", "Persuadable": "#F9A825", "Unlikely": "#E74C3C"}, title="Prediction Confidence")
            fig.update_layout(height=350, yaxis_title="Probability (%)", xaxis_title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            if pred_label == "Likely Adopter":
                st.markdown('<div class="strategy-box"><b>Strategy: Direct Conversion</b> — Push for sign-up with 14-day Premium trial. Highlight AI Savings & Predictive Alerts.</div>', unsafe_allow_html=True)
            elif pred_label == "Persuadable":
                st.markdown('<div class="warning-box"><b>Strategy: Nurture & Educate</b> — Send educational content, offer free tier, use social proof to build trust.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-box"><b>Strategy: Low Priority</b> — Add to awareness campaigns. Retarget in 3–6 months with brand-building content.</div>', unsafe_allow_html=True)

    else:
        st.markdown("### Upload New Customer Data (CSV)")
        st.caption("🔎 Upload a CSV with columns Q1–Q24 matching the survey structure. The system predicts each person's adoption class and recommends a marketing strategy.")

        template_cols = ["Q1_Age_Group", "Q2_Marital_Dependent_Status", "Q3_City_Tier", "Q4_Financial_Dependents", "Q5_Employment_Status",
            "Q6_Monthly_Income", "Q7_Income_Stability", "Q8_Monthly_Expenditure", "Q9_Top_Spending_Categories", "Q10_Impulse_Purchase_Frequency",
            "Q11_Spending_Triggers", "Q12_Active_Subscriptions", "Q13_Statement_Review_Frequency", "Q14_Financial_Goals", "Q15_Savings_Percentage",
            "Q16_Financial_Challenges", "Q17_Financial_Confidence", "Q18_Financial_Stress_Level", "Q19_Budget_Behavior", "Q20_Finance_App_Usage",
            "Q21_Financial_Literacy", "Q22_Digital_Comfort_Level", "Q23_Preferred_Features", "Q24_Willingness_To_Pay"]
        st.download_button("📥 Download CSV Template", pd.DataFrame(columns=template_cols).to_csv(index=False), "BudgetLife_Template.csv", "text/csv")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.success(f"Uploaded {len(new_data)} records!")
                st.dataframe(new_data.head(), use_container_width=True)

                if st.button("🔮 Run Predictions", use_container_width=True):
                    X_new = preprocess_new_customer(new_data, train_columns)
                    X_new_sc = scaler.transform(X_new)
                    predictions = rf_model.predict(X_new_sc)
                    probabilities = rf_model.predict_proba(X_new_sc)
                    pred_labels = le.inverse_transform(predictions)

                    results_df = new_data.copy()
                    results_df["Predicted_Category"] = pred_labels
                    for i, cls in enumerate(le.classes_):
                        results_df[f"Prob_{cls}"] = (probabilities[:, i] * 100).round(2)
                    strategy_map = {"Likely Adopter": "Direct Conversion", "Persuadable": "Nurture & Educate", "Unlikely": "Low Priority"}
                    results_df["Strategy"] = results_df["Predicted_Category"].map(strategy_map)

                    pred_summary = pd.Series(pred_labels).value_counts()
                    c1, c2, c3 = st.columns(3)
                    for i, cls in enumerate(le.classes_):
                        count = pred_summary.get(cls, 0)
                        [c1, c2, c3][i].metric(cls, f"{count} ({count/len(new_data)*100:.1f}%)")

                    fig = px.pie(values=pred_summary.values, names=pred_summary.index, color=pred_summary.index,
                               color_discrete_map={"Likely Adopter": "#27AE60", "Persuadable": "#F9A825", "Unlikely": "#E74C3C"}, title="Prediction Distribution", hole=0.4)
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(results_df, use_container_width=True, height=400)
                    st.download_button("📥 Download Results", results_df.to_csv(index=False), "BudgetLife_Predictions.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Ensure your CSV matches the template format.")
