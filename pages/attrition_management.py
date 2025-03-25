import streamlit as st
import pandas as pd

# Title of the App
st.markdown(
    "<h2 style='text-align: center; color: darkblue;'>📌 Employee Attrition Risk Management 📌</h2>", 
    unsafe_allow_html=True
)
st.write("")
st.write("")

# File Upload Section
uploaded_file = st.file_uploader("📂 Upload Employee Data CSV", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Check if required columns exist
    if 'Attrition_rate' in df.columns and 'Employee_ID' in df.columns:
        st.markdown(
            "<h3 style='color: darkblue;'>Attrition Rate Categorization</h3>", 
            unsafe_allow_html=True
        )

        # Define attrition risk categories
        def categorize_attrition(rate):
            if rate <= 0.33:
                return "Low Risk"
            elif rate <= 0.66:
                return "Medium Risk"
            else:
                return "High Risk"

        df["Attrition_Category"] = df["Attrition_rate"].apply(categorize_attrition)

        # Display categorized data
        st.write(df[["Employee_ID", "Attrition_rate", "Attrition_Category"]])


        st.write("")
        # Count of employees in each category
        category_counts = df["Attrition_Category"].value_counts()
        st.bar_chart(category_counts)

        # Retrieve Employee IDs & Attrition Rate of High Attrition Rate Employees
        high_risk_employees = df[df["Attrition_Category"] == "High Risk"][["Employee_ID", "Attrition_rate"]]

        # Display high-risk employees
        st.markdown(
            "<h4 style='color: red;'>⚠️ High Attrition Risk Employees</h4>", 
            unsafe_allow_html=True
        )
        if not high_risk_employees.empty:
            st.dataframe(high_risk_employees)  # Display as table with both Employee_ID & Attrition_rate
        else:
            st.info("✅ No employees fall under the High Risk category.")

        # ========== 🔍 Search Employee by ID ==========
        st.write("") 
        st.markdown(
            "<h4 style='color: black;'>🔍 Search Employee Data by Employee ID</h4>", 
            unsafe_allow_html=True
        )
        st.write("") 

        if "Employee_ID" in df.columns:
            employee_id = st.text_input("Enter Employee ID to Search", "").strip()

            if st.button("Search"):
                if employee_id:  
                    employee_data = df[df['Employee_ID'].astype(str) == employee_id]  
                    st.write("") 

                    if not employee_data.empty:
                        st.success(f"✅ Employee Found! Details for Employee ID: {employee_id}")
                        st.dataframe(employee_data)  

        # Define management strategies
        strategies = {
            "✅ Low Risk": [
                "Recognize and reward contributions.",
                "Offer career development programs.",
                "Provide incentives such as bonuses and promotions.",
            ],
            "⚠️ Medium Risk": [
                "Adjust salaries based on market rates.",
                "Implement flexible work schedules.",
                "Offer training and leadership programs.",
                "Conduct surveys to identify employee concerns.",
            ],
            "🚨 High Risk": [
                "Increase pay and benefits if below market standards.",
                "Allow remote/hybrid work to reduce commuting stress.",
                "Expedite salary increments and promotions.",
                "Reduce excessive overtime and prevent burnout.",
                "Conduct personalized retention meetings.",
            ],
        }
        st.write("") 
        st.write("") 
        st.markdown(
            "<h3 style='color: darkblue;'>📌 Management Strategies Based on Risk Levels</h3>", 
            unsafe_allow_html=True
        )
        
        for category, actions in strategies.items():
            st.markdown(f"<h4 style='color: black;'>{category} Employees</h4>", unsafe_allow_html=True)
            for action in actions:
                st.markdown(f"- {action}")

    else:
        st.error("⚠️ The uploaded file must contain both 'Employee_ID' and 'Attrition_rate' columns.")
