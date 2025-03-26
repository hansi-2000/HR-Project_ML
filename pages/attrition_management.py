import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.markdown("<h2 style='text-align: center; color: darkblue;'>📌 Employee Attrition Risk Management 📌</h2>", unsafe_allow_html=True)
st.write(" ")

# File Upload Section
uploaded_file = st.file_uploader("📂 Upload Employee Data CSV", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)

    # Ensure required columns exist
    required_columns = [
        'Employee_ID', 'Attrition_rate', 'Age', 'Time_of_service', 'Time_since_Salary_Increment',
        'Distance_from_Home', 'Workload_Index', 'Weekly_Over_Time', 'Decision_skill_possess', 
        'Post_Level', 'Yearly_Trainings', 'Pay_Scale', 'Education_Level', 'Growth_Rate', 
        'Work_Life_Balance', 'Compensation_and_Benefits'
    ]

    if all(col in df.columns for col in required_columns):

        # Categorize Attrition
        def categorize_attrition(rate):
            if rate <= 0.33:
                return "Low Risk"
            elif rate <= 0.66:
                return "Medium Risk"
            else:
                return "High Risk"

        df["Attrition_Category"] = df["Attrition_rate"].apply(categorize_attrition)

        # **1️⃣ Employee Search (Separate Feature)**
        st.markdown("<h3 style='color: darkblue;'>🔍 Search Employee Data</h3>", unsafe_allow_html=True)
        employee_id = st.text_input("Enter Employee ID to Search", "").strip()

        if st.button("Search"):
            if employee_id:
                employee_data = df[df['Employee_ID'].astype(str) == employee_id]

                if not employee_data.empty:
                    st.success(f"✅ Employee Found! Details for Employee ID: {employee_id}")
                    st.dataframe(employee_data)
                else:
                    st.error("⚠️ Employee not found!")

        st.write("---")  # Divider

        # **2️⃣ Attrition Measures (Separate Feature)**
        st.markdown("<h3 style='color: darkblue;'>📊 Attrition Risk Measures</h3>", unsafe_allow_html=True)

        # Show Attrition Rate Distribution
        st.write(df[["Employee_ID", "Attrition_rate", "Attrition_Category"]])

        # Show Bar Chart of Attrition Categories
        category_counts = df["Attrition_Category"].value_counts()
        # Set a modern style
        sns.set_style("whitegrid")

        # Create a figure
        fig, ax = plt.subplots(figsize=(9, 6))

        # Define colors for different categories
        colors = ["#FF5753", "#FFC100", "#3598DB"]  # High Risk (Red), Medium Risk (Yellow), Low Risk (Blue)

        # Create a Seaborn bar plot
        bars = sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors, ax=ax)

        # Add text labels above bars
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, height, f'{int(height)}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='black'
            )

        # Customize appearance
        ax.set_ylabel("Number of Employees", fontsize=12)
        ax.set_xlabel("Attrition Risk Category", fontsize=12)
        ax.set_title("📊 Employee Attrition Risk Distribution", fontsize=14, fontweight="bold")
        st.write(" ")
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        # Show the plot in Streamlit
        st.pyplot(fig)

        st.write("---")  # Divider

        # **3️⃣ Management Strategies (Separate Feature)**
        st.markdown("<h3 style='color: darkblue;'>📌 Management Strategies for Employee Categories</h3>", unsafe_allow_html=True)

        st.markdown("##### ✅ Low-Risk Employees (Attrition rate < 0.33)")
        st.markdown("""
        - Recognize and reward contributions.
        - Offer career development programs.
        - Provide performance-based incentives.
        """)

        st.markdown("##### ⚠️ Medium-Risk Employees (0.33 < Attrition rate < 0.66)")
        st.markdown("""
        - Adjust salaries based on market rates.
        - Implement flexible work schedules.
        - Offer leadership and training programs.
        """)

        st.markdown("##### 🚨 High-Risk Employees (Attrition rate > 0.66)")
        st.markdown("""
        - Increase pay and benefits if below market standards.
        - Reduce workload stress and overtime.
        - Provide hybrid/remote work options.
        - Conduct retention-focused meetings.
        """)

        st.write("---")  # Divider

        # **4️⃣ High-Risk Employee Management Feature**
        st.markdown("<h3 style='color: darkblue;'>⚠️ Manage High-Risk Employees</h3>", unsafe_allow_html=True)

        # Retrieve High-Risk Employees
        high_risk_employees = df[df["Attrition_Category"] == "High Risk"][["Employee_ID", "Attrition_rate"]]

        st.markdown("<h5 style='color: red;'>High Attrition Risk Employees</h5>", unsafe_allow_html=True)
        if not high_risk_employees.empty:
            st.dataframe(high_risk_employees)
        else:
            st.info("✅ No employees fall under the High Risk category.")

        st.write("---")  # Divider

        selected_high_risk_employee = st.selectbox("Select a High-Risk Employee", high_risk_employees["Employee_ID"] if not high_risk_employees.empty else [])

        if selected_high_risk_employee:
            employee_data = df[df["Employee_ID"] == selected_high_risk_employee]

            # Extract attrition rate and default feature values
            attrition_rate = employee_data['Attrition_rate'].values[0]
            default_values = {col: employee_data[col].values[0] for col in required_columns if col != 'Employee_ID'}

            st.markdown(f"<h6>🔹 Current Attrition Rate: <span style='color: darkblue;'>{attrition_rate:.4f}</span></h6>", unsafe_allow_html=True)
            st.write(" ")
            st.write(" ")

            # **Sliders for Impactful Attributes**
            st.markdown("<h5 style='color: black;'>🔧 Adjust Key Features with recommendation (Increase/ Decrease)</h5>", unsafe_allow_html=True)
            st.write(" ")

            updated_values = {}
            updated_values["Distance_from_Home"] = st.slider("🏡 Distance from Home (Decrease)", 0, 75, int(default_values["Distance_from_Home"]))
            updated_values["Workload_Index"] = st.slider("🔹 Workload Index (Decrease)", 1, 5, int(default_values["Workload_Index"]))
            updated_values["Weekly_Over_Time"] = st.slider("⌛ Weekly Overtime - hours (Decrease)", 1, 10, int(default_values["Weekly_Over_Time"]))
            updated_values["Work_Life_Balance"] = st.slider("⚖️ Work-Life Balance (Increase)", 1, 5, int(default_values["Work_Life_Balance"]))
            updated_values["Pay_Scale"] = st.slider("💰 Pay Scale (Increase)", 1, 10, int(default_values["Pay_Scale"]))
            updated_values["Compensation_and_Benefits"] = st.slider("📌 Compensation & Benefits (Increase)", 1, 5, int(default_values["Compensation_and_Benefits"]))
            updated_values["Growth_Rate"] = st.slider("📈 Growth Rate (Increase)", 0, 5, int(default_values["Growth_Rate"]))
            updated_values["Yearly_Trainings"] = st.slider("📈 Yearly Trainings Attended (Increase)", 0, 5, int(default_values["Yearly_Trainings"]))

            # **Calculate Adjusted Attrition Rate**
            correlation_values = {
            "Distance_from_Home": 0.010326, "Work_Life_Balance": -0.015774, 
            "Pay_Scale": -0.005127, "Compensation_and_Benefits": -0.027160, "Workload_Index": 0.008626, 
            "Weekly_Over_Time": 0.003030, "Growth_Rate": -0.010573, "Yearly_Trainings": -0.01875
            }

            adjusted_attrition = attrition_rate  # Start with original attrition rate

            for feature, change in updated_values.items():
                initial_value = default_values[feature]
                diff = change - initial_value
                adjusted_attrition += diff * correlation_values[feature]

            adjusted_attrition = np.clip(adjusted_attrition, 0, 1)  # Keep in range [0,1]
            
            st.write(" ")
            st.write(" ")
            st.markdown(f"<h5>🔹 Current Attrition Rate: <span style='color: darkblue;'>{attrition_rate:.4f}</span></h5>", unsafe_allow_html=True)
            st.write(" ")
            # Display adjusted attrition rate
            st.markdown(f"<h5>🔹 Adjusted Attrition Rate: <span style='color: darkred;'>{adjusted_attrition:.4f}</span></h5>", unsafe_allow_html=True)

    else:
        st.error("⚠️ The uploaded file must contain all required columns.")
