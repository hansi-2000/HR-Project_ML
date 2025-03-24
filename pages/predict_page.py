import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
attrition_model = pickle.load(open('model.pkl', 'rb'))

def predict_attrition(model, data):
    """
    Function to make predictions using the trained regression model.
    """
    predictions = model.predict(data)
    return predictions

def main():
    #st.markdown('<div class="title-container"><h1 style="color:#333;font-size:40px;margin-bottom:18px">🔍 Employee Attrition Rate Prediction</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color:#008080; padding:12px; text-align:center;font-size:10px; border-radius:10px;">
        <h2 style="color:white; margin-bottom:10px;">🔍 Employee Attrition Rate Prediction</h2>
    </div>
""", unsafe_allow_html=True)
    
    
    # Sidebar input selection
    st.sidebar.header("Select Input Method")
    input_method = st.sidebar.radio("", ('📂 Upload CSV File', '✍️ Enter Employee Data Manually'))

    # CSV File Upload
    st.write(" ")
    st.write(" ")
    if input_method == '📂 Upload CSV File':
        uploaded_file = st.file_uploader("📂 Upload Employee Data (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File Uploaded Successfully!")
            
            # Display a preview of the dataset
            st.subheader("📌 Preview of Uploaded Data")
            st.write(df.head())

            # Define mappings for categorical columns
            decision_skill_mapping = {'Directive': 1, 'Analytical': 2, 'Conceptual': 3, 'Behavioral': 4}
            compensation_mapping = {'type0': 1.0, 'type1': 2.0, 'type2': 3.0, 'type3': 4.0, 'type4': 5.0}

            # Apply mapping before selecting features
            df['Decision_skill_possess'] = df['Decision_skill_possess'].astype(str).map(decision_skill_mapping)
            df['Compensation_and_Benefits'] = df['Compensation_and_Benefits'].astype(str).map(compensation_mapping)

            # Select relevant features
            feature_columns = ['Age', 'Education_Level', 'Decision_skill_possess', 'Time_of_service', 
                            'Time_since_Salary_Increment', 'Distance_from_Home', 'Workload_Index', 
                            'Pay_Scale', 'Compensation_and_Benefits', 'Post_Level', 'Growth_Rate',
                            'Yearly_Trainings', 'Weekly_Over_Time', 'Work_Life_Balance']
                    
            # Ensure correct data selection
            features = df.loc[:, feature_columns]

            # Convert all features to numeric (coerce invalid values)
            features = features.apply(pd.to_numeric, errors='coerce')

            valid_indexes = features.dropna().index
            
            # Drop rows with NaN values (optional)
            features = features.dropna()

            if features.isnull().any().any():
                st.error("Error: Some data could not be converted. Please check the dataset.")
            else:
                if st.button('Predict Attrition Rate'):
                    try:
                        predictions = predict_attrition(attrition_model, features.values)
                        df['Attrition Rate Prediction'] = np.nan
                        df.loc[valid_indexes, 'Attrition Rate Prediction'] = predictions
                        st.write("Predictions (Attrition Rate):")
                        st.write(df)

                        # Convert DataFrame to CSV for download
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Predictions", csv, "attrition_predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Error: {e}")

    
    elif input_method == '✍️ Enter Employee Data Manually':
        st.subheader("✍️ Enter Employee Details")
        st.write("") 
        # Manual input fields for each attribute
        Employee_ID = st.text_input("🔢 Employee ID:")
        st.write("") 
        Age = float(st.number_input("🎂 Age:", min_value=18, max_value=65))
        st.write("") 
        Education_Level = float(st.selectbox('🎓 Education Level:', ['1', '2', '3', '4', '5']))
        st.write("") 
        Post_Level = float(st.selectbox('📊 Post Level:', ['1', '2', '3', '4', '5']))
        st.write("") 
        Time_of_Service = float(st.number_input("🕰️ Time of Service:"))
        st.write("") 
        Distance_from_Home = float(st.number_input("🚗 Distance from Home:"))
        st.write("") 
        Work_Life_Balance = float(st.selectbox('⚖️ Work Life Balance:', ['1', '2', '3', '4', '5']))
        st.write("") 
        Growth_Rate = float(st.selectbox('📈 Growth Rate:', ['1', '2', '3', '4', '5']))
        st.write("") 
        Pay_Scale = float(st.slider('💰 Pay Scale:', 0.0, 10.0))
        st.write("") 
        Time_since_Salary_Increment = float(st.slider('📆 Time since Salary Increment:', 0.0, 6.0))
        st.write("") 
        Compensation_and_Benefits = float(st.slider('🏆 Compensation and Benefits:', 1.0, 5.0))
        st.write("") 
        Workload_Index = float(st.slider('📝 Workload Index:', 1.0, 5.0))
        st.write("") 
        Weekly_Over_Time = float(st.slider('⏳ Weekly Over Time:', 0.0, 10.0))
        st.write("") 
        Decision_skill_possess = float(st.selectbox('🧠 Decision Skill Level:', ['1', '2', '3', '4']))
        st.write("") 
        Yearly_Trainings = float(st.selectbox('📚 Yearly Trainings Attended:', ['1', '2', '3', '4', '5']))
        st.write("") 
        
        # Prepare input data for prediction
        user_inputs = [[Age, Education_Level, Post_Level, Time_of_Service, Distance_from_Home, Work_Life_Balance, 
                        Growth_Rate, Pay_Scale, Time_since_Salary_Increment, Compensation_and_Benefits, 
                        Workload_Index, Weekly_Over_Time, Decision_skill_possess, Yearly_Trainings]]
        
        st.markdown("""
            <style>
                div.stButton > button {
                    background-color: #008080; /* Teal color */
                    color: white; /* White text */
                    border-radius: 10px;
                    border: 2px solid #005f5f; /* Darker teal border */
                    padding: 10px 20px;
                    font-size: 16px;
                    font-weight: bold;
                    transition: 0.3s;
                }
                
                div.stButton > button:hover {
                    background-color: #005f5f; /* Darker teal on hover */
                    color: #ffffff;
                }
            </style>
        """, unsafe_allow_html=True)

        if st.button("Predict Attrition Rate"):
            # Check if any field is empty
            if not Employee_ID.strip():  # Ensure Employee ID is not just spaces
                st.error("❌ Please enter a valid Employee ID!")
            elif not Age:
                st.error("❌ Please enter Age!")
            elif not Time_of_Service:
                st.error("❌ Please enter Time of Service!")
            elif not Distance_from_Home:
                st.error("❌ Please enter Distance from Home!")
            elif not Pay_Scale:
                st.error("❌ Please enter Pay Scale!")
            elif not Time_since_Salary_Increment:
                st.error("❌ Please enter Time since Salary Increment!")
            elif not Compensation_and_Benefits:
                st.error("❌ Please enter Compensation and Benefits!")
            elif not Workload_Index:
                st.error("❌ Please enter Workload Index!")
            elif not Weekly_Over_Time:
                st.error("❌ Please enter Weekly Over Time!")
            else:
                try:
                    # Make the prediction using the regression model
                    prediction = predict_attrition(attrition_model, user_inputs)
                    st.write(f"Predicted Attrition Rate: {prediction[0]:.2f}")
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
