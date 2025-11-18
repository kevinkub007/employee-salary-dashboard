import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import kagglehub
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Employee Salary Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("alizabrand/employee-salary-analysis-dataset")
    csv_files = list(Path(path).glob("*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        return df
    return None

# Header with image
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
with col2:
    st.title("💼 Employee Salary Analytics Dashboard")
    st.markdown("**Comprehensive analysis of employee compensation and demographics**")

# Load data
with st.spinner("Loading data..."):
    df = load_data()

if df is None:
    st.error("Could not load the dataset. Please check the data source.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Detect available columns
available_cols = df.columns.tolist()

# Try to identify key columns (flexible to different column names)
salary_col = next((col for col in available_cols if 'salary' in col.lower()), None)
dept_col = next((col for col in available_cols if 'department' in col.lower() or 'dept' in col.lower()), None)
gender_col = next((col for col in available_cols if 'gender' in col.lower() or 'sex' in col.lower()), None)
age_col = next((col for col in available_cols if 'age' in col.lower()), None)
experience_col = next((col for col in available_cols if 'experience' in col.lower() or 'years' in col.lower()), None)
education_col = next((col for col in available_cols if 'education' in col.lower() or 'degree' in col.lower()), None)

# Add filters based on available columns
if dept_col and dept_col in df.columns:
    departments = ['All'] + sorted(df[dept_col].dropna().unique().tolist())
    selected_dept = st.sidebar.multiselect("Department", departments, default=['All'])
    if 'All' not in selected_dept:
        df = df[df[dept_col].isin(selected_dept)]

if gender_col and gender_col in df.columns:
    genders = ['All'] + sorted(df[gender_col].dropna().unique().tolist())
    selected_gender = st.sidebar.multiselect("Gender", genders, default=['All'])
    if 'All' not in selected_gender:
        df = df[df[gender_col].isin(selected_gender)]

if education_col and education_col in df.columns:
    education_levels = ['All'] + sorted(df[education_col].dropna().unique().tolist())
    selected_education = st.sidebar.multiselect("Education", education_levels, default=['All'])
    if 'All' not in selected_education:
        df = df[df[education_col].isin(selected_education)]

st.sidebar.markdown("---")
st.sidebar.info("📊 Dashboard created with Streamlit & Plotly")

# Key Metrics
st.markdown("## 📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Employees", f"{len(df):,}")
with col2:
    if salary_col:
        avg_salary = df[salary_col].mean()
        st.metric("Average Salary", f"${avg_salary:,.0f}")
with col3:
    if salary_col:
        median_salary = df[salary_col].median()
        st.metric("Median Salary", f"${median_salary:,.0f}")
with col4:
    if dept_col:
        dept_count = df[dept_col].nunique()
        st.metric("Departments", dept_count)

# Main visualizations
st.markdown("---")

# Row 1: Two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💰 Salary Distribution")
    if salary_col:
        fig = px.histogram(df, x=salary_col, nbins=30, 
                          color_discrete_sequence=['#1f77b4'],
                          labels={salary_col: 'Salary'},
                          title="")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🏢 Employees by Department")
    if dept_col:
        dept_counts = df[dept_col].value_counts()
        fig = px.pie(values=dept_counts.values, names=dept_counts.index,
                    hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig, use_container_width=True)

# Row 2: Two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Average Salary by Department")
    if salary_col and dept_col:
        avg_sal_dept = df.groupby(dept_col)[salary_col].mean().sort_values(ascending=True)
        fig = px.bar(x=avg_sal_dept.values, y=avg_sal_dept.index,
                    orientation='h',
                    labels={'x': 'Average Salary', 'y': 'Department'},
                    color=avg_sal_dept.values,
                    color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 👥 Gender Distribution")
    if gender_col:
        gender_counts = df[gender_col].value_counts()
        fig = px.bar(x=gender_counts.index, y=gender_counts.values,
                    labels={'x': 'Gender', 'y': 'Count'},
                    color=gender_counts.index,
                    color_discrete_sequence=['#ff7f0e', '#2ca02c', '#9467bd'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# Row 3: Full width
st.markdown("### 📉 Detailed Salary Analysis")

if salary_col and age_col:
    fig = px.scatter(df, x=age_col, y=salary_col,
                    color=dept_col if dept_col else None,
                    size=experience_col if experience_col else None,
                    hover_data=df.columns,
                    labels={age_col: 'Age', salary_col: 'Salary'},
                    color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
elif salary_col and experience_col:
    fig = px.scatter(df, x=experience_col, y=salary_col,
                    color=dept_col if dept_col else None,
                    hover_data=df.columns,
                    labels={experience_col: 'Experience (Years)', salary_col: 'Salary'},
                    color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Data table
st.markdown("---")
st.markdown("### 📋 Raw Data")
st.dataframe(df, use_container_width=True, height=400)

# Download button
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_employee_data.csv",
    mime="text/csv",
)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d;'>Built with ❤️ using Streamlit | Data Analytics Dashboard</div>",
    unsafe_allow_html=True
)
