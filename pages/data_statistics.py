import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define mappings for categorical columns
DECISION_SKILL_MAPPING = {'Directive': 1, 'Analytical': 2, 'Conceptual': 3, 'Behavioral': 4}
COMPENSATION_MAPPING = {'type0': 1.0, 'type1': 2.0, 'type2': 3.0, 'type3': 4.0, 'type4': 5.0}

FEATURE_COLUMNS = ['Age', 'Education_Level', 'Post_Level', 'Time_of_Service', 'Distance_from_Home', 'Work_Life_Balance', 
                        'Growth_Rate', 'Pay_Scale', 'Time_since_Salary_Increment', 'Compensation_and_Benefits', 
                        'Workload_Index', 'Weekly_Over_Time', 'Decision_skill_possess', 'Yearly_Trainings']

# Streamlit Page Config
st.set_page_config(page_title="Employee Data Statistics", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        background-color: #008080; 
        padding: 8px; 
        text-align: center; 
        border-radius: 10px;
    }
    .main-title h2 {
        color: white;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #008080; 
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title"><h2>📊 Statistics of Employee Data</h2></div>', unsafe_allow_html=True)
st.write("")  # Space

# File uploader
uploaded_file = st.file_uploader("📂 Upload Employee Data (CSV)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, dtype={'Employee_ID': str})  # Ensure Employee_ID is string
        st.success("✅ File Uploaded Successfully!")

        # Display a preview of the dataset
        st.write("") 
        st.write("") 
        st.subheader("1. Data Preview")
        st.dataframe(df.head())  

        # Apply mapping before selecting features
        df['Decision_skill_possess'] = df['Decision_skill_possess'].astype(str).map(DECISION_SKILL_MAPPING)
        df['Compensation_and_Benefits'] = df['Compensation_and_Benefits'].astype(str).map(COMPENSATION_MAPPING)

        # Select relevant features
        available_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
        
        if not available_columns:
            st.error("⚠️ No required columns found in the dataset!")
            st.stop()
        
        features = df[available_columns]
        features = features.apply(pd.to_numeric, errors='coerce', downcast='integer', axis=1)  
        features = features.dropna()  

        # Show the cleaned data
        st.write("") 
        st.subheader("2. Processed Data (After Cleaning)")
        st.dataframe(features.head())
        
        st.write("") 
        st.subheader("3. Data Overview")

        # Subheadings in smaller font
        st.markdown("#### 🔹 Dataset Summary")
        st.dataframe(df.describe().style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]))

        st.markdown("#### 🔹 Missing Values")
        missing_values = df.isnull().sum().reset_index()
        missing_values.columns = ['Column', 'Missing Values']

        # Use st.markdown with CSS for narrow table width
        st.markdown("""
        <style>
            .small-table {
                width: 20%;  /* Adjust width as needed */
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="small-table">', unsafe_allow_html=True)
        st.table(missing_values)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### 🔹 Unique Values Per Column")
        unique_values = df.nunique().reset_index()
        unique_values.columns = ['Column', 'Unique Values']

        st.markdown('<div class="small-table">', unsafe_allow_html=True)
        st.table(unique_values)
        st.markdown('</div>', unsafe_allow_html=True)



        # ========== 🔍 Search Employee by ID ==========
        st.write("") 
        st.subheader("3. Search Employee Data by Employee ID")
        st.write("") 

        if "Employee_ID" in df.columns:
            employee_id = st.text_input("Enter Employee ID to Search", "").strip()
            
            if st.button("Search"):
                if employee_id:  
                    employee_data = df[df['Employee_ID'] == employee_id]  # No conversion needed
                    st.write("") 
                    
                    if not employee_data.empty:
                        st.success(f"✅ Employee Found! Details for Employee ID: {employee_id}")
                        st.dataframe(employee_data)  

                     
        st.write("") 
        st.subheader("4. Generate Graphs")
        st.write("🔹 Histogram representation for features")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        col = st.selectbox("Select Column for Histogram", num_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
        
        st.write("") 
        st.write("") 
        
        number_cols = ['Age','Time_of_Service', 'Distance_from_Home','Time_since_Salary_Increment', 
                        'Weekly_Over_Time']
        
        st.write("🔹 Boxplot representations for outliers")
        col = st.selectbox("Select Column for outliers", number_cols)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)
 
        st.write("") 
        st.write("") 
        
        categorical_cols = ['Education_Level', 'Post_Level', 'Distance_from_Home', 'Work_Life_Balance', 
                        'Growth_Rate', 'Pay_Scale', 'Compensation_and_Benefits', 
                        'Workload_Index', 'Decision_skill_possess', 'Yearly_Trainings']
        
        st.write("🔹 Pie chart for categorical proportion")
        col_1 = st.selectbox("Select categorical column", categorical_cols)
        fig, ax = plt.subplots()
        df[col_1].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_title(f"Proportion of {col_1}")
        st.pyplot(fig)
       
        st.write("") 
        st.write("") 
        
        # User selects attributes for X and Y axes
        st.write("🔹 Relationship between two feature fields") 
        st.write("") 
        st.write("📉 Scatter Plot (For Numerical Features)")
        x_col = st.selectbox("Select X-Axis Attribute", available_columns[1:])
        y_col = st.selectbox("Select Y-Axis Attribute", available_columns[1:])
        
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
        ax.set_title(f"{y_col} vs {x_col}")
        st.pyplot(fig)
        st.write("") 
        
        st.write("📉 Bar Chart (For Categorical vs Numerical)")
        cat_cols = df.select_dtypes(include=['object']).columns
        cat_col = st.selectbox("Select Categorical Column", cat_cols)
        num_col = st.selectbox("Select Numerical Column", num_cols)

        fig, ax = plt.subplots()
        df.groupby(cat_col)[num_col].mean().plot(kind='bar', ax=ax, color='teal')
        ax.set_title(f"Average {num_col} by {cat_col}")
        st.pyplot(fig)
        st.write("") 
        
        st.write("📉 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

else:
    st.info("📤 Please upload a dataset to proceed.")
