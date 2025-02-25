import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open("student_lr_pred_model.pkl", "rb") as file:
        model, scaler, le = pickle.load(file)
    return model , scaler, le

def preprocessing_input(data, scaler, le):
    df = pd.DataFrame([data])
    df["Extracurricular Activities"] = le.transform(df["Extracurricular Activities"])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model , scaler, le = load_model()
    processed_data = preprocessing_input(data,scaler, le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student performance predictions")
    st.write("Enter your data to get the prediction of your performance")

    Hours_Studied = st.number_input("Enter Hours studied", min_value= 1 , max_value=10, value=5)
    Previous_Scores = st.number_input("Enter Previoud score ", min_value= 0 , max_value=100, value=50)
    Extracurricular_Activities = st.selectbox("Extra caricular value", ['Yes', 'No'])
    Sleep_Hours = st.number_input("Sleep Hours", min_value= 5 , max_value=24, value=6)
    Sample_Paper_solved = st.number_input("Number of question paper solved", min_value= 0 , max_value=1000, value=8)

    if st.button("Predict your score"):
        user_data = {

            'Hours Studied' :Hours_Studied ,
            'Previous Scores':Previous_Scores,
            'Extracurricular Activities':Extracurricular_Activities,
            'Sleep Hours': Sleep_Hours,
            'Sample Question Papers Practiced':Sample_Paper_solved

        }

        prediction = predict_data(user_data)
        st.success(f"Your projected score is {prediction} ")
        st.write(" Sugestion Based on Model calculator ")
        st.write(" your performance will increase if you try to solve more sample papers")
        


if __name__ == "__main__":
    main()





    