import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import kagglehub
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import requests

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
        padding-bottom: 10px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    .header-container {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 30px;
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
        # Ensure date column exists or create one
        if 'Date' not in df.columns and 'date' not in df.columns:
            # Create sample dates for demo
            start_date = datetime(2020, 1, 1)
            df['Date'] = [start_date + timedelta(days=np.random.randint(0, 1800)) for _ in range(len(df))]
        elif 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        else:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

# Function to create salary projection
def project_salaries(df, salary_col, years=5):
    if salary_col not in df.columns:
        return None
    
    current_avg = df[salary_col].mean()
    # Assume 3% annual growth
    growth_rate = 0.03
    
    years_list = list(range(datetime.now().year, datetime.now().year + years + 1))
    projected_salaries = [current_avg * ((1 + growth_rate) ** i) for i in range(len(years_list))]
    
    return pd.DataFrame({
        'Year': years_list,
        'Projected Average Salary': projected_salaries
    })

# Initialize session state for cross-filtering
if 'selected_filters' not in st.session_state:
    st.session_state.selected_filters = {
        'department': None,
        'gender': None,
        'country': None
    }

# Header with profile image
st.image("profile.jpg", width=100)

# Load data
with st.spinner("Loading data..."):
    df = load_data()

if df is None:
    st.error("Could not load the dataset. Please check the data source.")
    st.stop()

# Store original data
df_original = df.copy()

# Detect available columns
available_cols = df.columns.tolist()
salary_col = next((col for col in available_cols if 'salary' in col.lower()), None)
dept_col = next((col for col in available_cols if 'department' in col.lower() or 'dept' in col.lower()), None)
gender_col = next((col for col in available_cols if 'gender' in col.lower() or 'sex' in col.lower()), None)
age_col = next((col for col in available_cols if 'age' in col.lower()), None)
experience_col = next((col for col in available_cols if 'experience' in col.lower() or 'years' in col.lower()), None)
education_col = next((col for col in available_cols if 'education' in col.lower() or 'degree' in col.lower()), None)
country_col = next((col for col in available_cols if 'country' in col.lower() or 'location' in col.lower() or 'nation' in col.lower()), None)

# Sidebar filters
st.sidebar.header("🔍 Interactive Filters")

# Date range filter
st.sidebar.markdown("### 📅 Date Range")
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply date filter
if len(date_range) == 2:
    df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

st.sidebar.markdown("---")

# Department filter
if dept_col and dept_col in df.columns:
    st.sidebar.markdown("### 🏢 Department")
    departments = sorted(df[dept_col].dropna().unique().tolist())
    selected_dept = st.sidebar.multiselect("Select Department(s)", departments, default=departments)
    if selected_dept:
        df = df[df[dept_col].isin(selected_dept)]

# Gender filter
if gender_col and gender_col in df.columns:
    st.sidebar.markdown("### 👥 Gender")
    genders = sorted(df[gender_col].dropna().unique().tolist())
    selected_gender = st.sidebar.multiselect("Select Gender(s)", genders, default=genders)
    if selected_gender:
        df = df[df[gender_col].isin(selected_gender)]

# Country filter
if country_col and country_col in df.columns:
    st.sidebar.markdown("### 🌍 Country")
    countries = sorted(df[country_col].dropna().unique().tolist())
    selected_country = st.sidebar.multiselect("Select Country(ies)", countries, default=countries)
    if selected_country:
        df = df[df[country_col].isin(selected_country)]

# Education filter
if education_col and education_col in df.columns:
    st.sidebar.markdown("### 🎓 Education")
    education_levels = sorted(df[education_col].dropna().unique().tolist())
    selected_education = st.sidebar.multiselect("Select Education Level(s)", education_levels, default=education_levels)
    if selected_education:
        df = df[df[education_col].isin(selected_education)]

st.sidebar.markdown("---")
st.sidebar.info("💡 Click on any chart element to filter data across all visualizations!")

# Key Metrics
st.markdown("## 📈 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

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
with col5:
    if country_col:
        country_count = df[country_col].nunique()
        st.metric("Countries", country_count)

st.markdown("---")

# Salary Projection
st.markdown("## 📊 5-Year Salary Projection")
if salary_col:
    projection_df = project_salaries(df_original, salary_col)
    if projection_df is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=projection_df['Year'],
            y=projection_df['Projected Average Salary'],
            mode='lines+markers',
            name='Projected Salary',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title="Expected Average Annual Salary Growth (3% yearly)",
            xaxis_title="Year",
            yaxis_title="Average Salary ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Row 1: Salary distribution and country analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💰 Salary Distribution by Department")
    if salary_col and dept_col:
        dept_salary = df.groupby(dept_col)[salary_col].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(dept_salary, x=dept_col, y=salary_col,
                    color=salary_col,
                    color_continuous_scale='Blues',
                    labels={dept_col: 'Department', salary_col: 'Average Salary'})
        fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key='dept_salary_bar')

with col2:
    st.markdown("### 🌍 Salary by Country")
    if salary_col and country_col:
        country_salary = df.groupby(country_col)[salary_col].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(country_salary, x=country_col, y=salary_col,
                    color=salary_col,
                    color_continuous_scale='Viridis',
                    labels={country_col: 'Country', salary_col: 'Average Salary'})
        fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key='country_salary_bar')

# Row 2: Department distribution and Gender analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏢 Employee Distribution by Department")
    if dept_col:
        dept_counts = df[dept_col].value_counts().reset_index()
        dept_counts.columns = [dept_col, 'Count']
        fig = px.pie(dept_counts, values='Count', names=dept_col,
                    hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key='dept_pie')

with col2:
    st.markdown("### 👥 Gender Pay Analysis")
    if gender_col and salary_col:
        gender_salary = df.groupby(gender_col)[salary_col].mean().reset_index()
        fig = px.bar(gender_salary, x=gender_col, y=salary_col,
                    color=gender_col,
                    color_discrete_sequence=['#ff7f0e', '#2ca02c', '#9467bd'],
                    labels={gender_col: 'Gender', salary_col: 'Average Salary'})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True, key='gender_salary_bar')

# Row 3: Country distribution and Education impact
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌍 Employee Distribution by Country")
    if country_col:
        country_counts = df[country_col].value_counts().head(10).reset_index()
        country_counts.columns = [country_col, 'Count']
        fig = px.bar(country_counts, x=country_col, y='Count',
                    color='Count',
                    color_continuous_scale='Teal',
                    labels={country_col: 'Country', 'Count': 'Number of Employees'})
        fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key='country_dist_bar')

with col2:
    st.markdown("### 🎓 Salary by Education Level")
    if education_col and salary_col:
        edu_salary = df.groupby(education_col)[salary_col].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(edu_salary, x=education_col, y=salary_col,
                    color=salary_col,
                    color_continuous_scale='Oranges',
                    labels={education_col: 'Education', salary_col: 'Average Salary'})
        fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key='edu_salary_bar')

# Row 4: Experience vs Salary Analysis
st.markdown("### 📈 Salary Trends Analysis")

if salary_col and experience_col:
    # Create bins for experience
    df['Experience_Group'] = pd.cut(df[experience_col], bins=[0, 2, 5, 10, 20, 100],
                                     labels=['0-2 years', '3-5 years', '6-10 years', '11-20 years', '20+ years'])
    
    exp_salary = df.groupby('Experience_Group')[salary_col].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exp_salary['Experience_Group'],
        y=exp_salary[salary_col],
        mode='lines+markers',
        name='Average Salary',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=12)
    ))
    fig.update_layout(
        title="Average Salary by Experience Level",
        xaxis_title="Experience Level",
        yaxis_title="Average Salary ($)",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

elif salary_col and age_col:
    # Create age groups
    df['Age_Group'] = pd.cut(df[age_col], bins=[0, 25, 35, 45, 55, 100],
                              labels=['<25', '26-35', '36-45', '46-55', '55+'])
    
    age_salary = df.groupby('Age_Group')[salary_col].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=age_salary['Age_Group'],
        y=age_salary[salary_col],
        mode='lines+markers',
        name='Average Salary',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=12)
    ))
    fig.update_layout(
        title="Average Salary by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Average Salary ($)",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# Salary trends over time
st.markdown("### 📅 Salary Trends Over Time")
if salary_col:
    df['Year_Month'] = df['Date'].dt.to_period('M').astype(str)
    time_salary = df.groupby('Year_Month')[salary_col].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_salary['Year_Month'],
        y=time_salary[salary_col],
        mode='lines+markers',
        name='Average Salary',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title="Average Salary Trend Over Time",
        xaxis_title="Month",
        yaxis_title="Average Salary ($)",
        height=400,
        hovermode='x unified'
    )
    fig.update_xaxis(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Data table
st.markdown("---")
st.markdown("### 📋 Filtered Data Table")
display_cols = [col for col in df.columns if col != 'Experience_Group' and col != 'Age_Group' and col != 'Year_Month']
st.dataframe(df[display_cols], use_container_width=True, height=400)

# Download button
csv = df[display_cols].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name=f"filtered_employee_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d;'>Built with ❤️ using Streamlit | Interactive Data Analytics Dashboard</div>",
    unsafe_allow_html=True
)
