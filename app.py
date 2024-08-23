import streamlit as st
import pandas as pd
import openai

# Set your OpenAI API key
openai.api_key = 'your_openai_key'

# Load data
data = pd.read_csv('healthcare_dataset.csv')

# Function to query patients based on a symptom
def query_patients_by_symptom(symptom):
    if 'Medical Condition' not in data.columns:
        raise ValueError("The dataset does not contain a 'Medical Condition' column.")
    
    # Filter patients with the specified symptom
    filtered_data = data[data['Medical Condition'].str.contains(symptom, case=False, na=False)]
    
    # Limit number of rows to avoid excessive token usage
    filtered_data = filtered_data.head(10)
    details = filtered_data[['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition', 'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider', 'Billing Amount', 'Room Number', 'Admission Type', 'Discharge Date', 'Medication', 'Test Results']]
    
    return details

# Function to generate a response using the chat model
def generate_response(symptom):
    patient_details = query_patients_by_symptom(symptom)
    
    if not patient_details.empty:
        # Convert details to string and limit context if necessary
        context = patient_details.to_string(index=False)
        if len(context) > 3000:
            context = context[:3000] + '...'
        prompt = f"Provide a detailed summary for the following patient details with {symptom}:\n{context}"
    else:
        prompt = f"No patients found with {symptom} in the dataset."
    
    # Call the chat model endpoint
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please summarize the patient details as accurately as possible."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500 #aster per need increase the token for testing i used 500 
        
    )
    
    return response.choices[0].message['content'].strip()

# Streamlit app
st.title('CareCraft AI - Healthcare Assistant')

symptom = st.text_input("Enter a symptom or condition:")

if st.button("Get Patient Details"):
    if symptom:
        response = generate_response(symptom)
        st.write(response)
    else:
        st.write("Please enter a symptom or condition to query.")
