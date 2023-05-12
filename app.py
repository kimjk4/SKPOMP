#Core Pkgs
import streamlit as st

#EDA Pkgs
import pandas as pd
import numpy as np
import math

#Utils
import os
import joblib
import hashlib

#passlib,bcrypt

#Data Viz Pckgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

feature_names_best = ['age_baseline', 'gender', 'laterality', 'length_1_1',
       'apd_us1', 'sfu_us1', 'max_dilation1']

Gender_dict = {"Male":1, "Female":2}
Laterality_dict = {"Left":1, "Right":2, "Bilateral":3}
feature_dict = {"Yes":1, "No":0}

def get_value(val,my_dict):
    return new_func(val, my_dict)

def new_func(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val):
    for key, value in my_dict.items():
        if val == key:
            return value
        

# Load ML models        
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

header_container = st.container()
stats_container = st.container()	

about_container1 = st.container()
about_container2 = st.container()	
about_container3 = st.container()
about_container4 = st.container()
about_container5 = st.container()

col_1, col_2, col_3 = st.columns(3)

col_4, col_5 = st.columns(2)
col_6, col_7 = st.columns(2)
col_8, col_9 = st.columns(2)
col_10, col_11 = st.columns(2)

activity = st.selectbox ('Select Activity', ['Prediction', 'About'])


with header_container:
    st.title('SK-POMP: SickKids Primary obstructive megaureter prediction')
    st.caption("This is a web app to predict the likelihood of identfying obstruction on MAG3 scan (defined as t1/2 > 20 minutes) based on an infant's baseline clinical and ultrasound characteristics.")
    st.caption('This model is currently in development. Further external validation is required before wide use in clinical decision making. Please use at your own risk.')

with stats_container:
    def main():
        
        if activity == 'Prediction':
            
            with col_1:
            
                st.subheader ('Clinical Characteristics')
                age_baseline = st.number_input ("Age (days)", 0, 30)
                gender = st.radio ("Sex", tuple(Gender_dict.keys()))
                laterality = st.radio ("Side of hydronephrosis", tuple(Laterality_dict.keys()))
                st.caption('If bilateral, ultrasound findings should be focused on the side with worse hydroureter.')

        
            with col_2:

                st.subheader ('Ultrasound findings')
                length_1_1 = st.number_input ("Length (mm)", 0, 110)
                apd_us1 = st.number_input ("AP diameter (mm)", 0, 45)
                sfu_us1 = st.number_input ("SFU Grading", 0, 4)
                max_dilation1 = st.number_input ("maximum ureter diameter (mm)", 0, 50)
            
            with col_3:

                feature_list = [age_baseline, get_value(gender, Gender_dict),  get_value(laterality, Laterality_dict), length_1_1, 
                                apd_us1, sfu_us1, max_dilation1]
                single_sample = np.array(feature_list).reshape(1,-1)

                model_choice = st.selectbox("Select Model", ["Calibrated logistic regression"])
                if st.button("Predict"):    
                    if model_choice == "Calibrated logistic regression":
                        loaded_model = load_model("calibratedlogmodel.pkl")
                        prediction = loaded_model.predict(single_sample)
                        proba = loaded_model.predict_proba(single_sample)

                    if prediction == 1:
                        st.success("The patient is likely to have obstruction on MAG3 scan.")
                    else:
                        st.success("The patient is unlikely to have obstruction on MAG3 scan.")

                    st.write("Prediction: ", prediction)
                    st.write("Probability of obstruction on MAG3: ", proba)

        if activity == 'About':
            
            with about_container1:
                st.subheader ('This is a prediction of primary obstructive megaureter for patients with hydroureter, developed at The Hospital for Sick Children (SickKids), Toronto, Ontario, Canada.')
                st.write ('The tool is based on 183 infants identified to have primary non-refluxing megaureter.')
                st.caption('Among infants identified to have hydronephrosis, those with primary non-refluxing megaureter accounts for the minority. Without a mercaptoacetyltriglycine-3 (MAG-3) diuretic renal scan, it is difficult to discern whether the cause of the megaureter is due to obstruction. Hence, we aim to develop a prediction model, specifically for the megaureter population, to predict the likelihood of detecting obstruction on MAG-3 scan based on clinical and ultrasound characteristics.')

            with about_container2:
                with col_4:

                    st.write('Using plot densities of the variables, we identified variables that had potential to differentiate patients who are likely to have obstruction')
                with col_5:
                    st.image('SFU.png')
                    st.caption('Example plot of SFU grading demonstrating potential to differentiate patients who are likely to have obstruction.')

            with about_container3:
                with col_6:
                    st.write('Following this, we developed a logistic regression model with L2 regularization and was able to calibrate it to provide better prediction.')
                with col_7:
                    st.image('calibration.png')
                    st.caption('Calibration curve of the logistic regression model (pre- and post-calibration).')

            with about_container4:
                with col_8:
                    st.write('The final model had a area under receiving operating characteristics curve (AUROC) of 0.817 and area under precision-recall curve (AUPRC) of 0.736 with f1 score of 0.700. The model had excellent negative predictive value (0.923) and specificity (0.857).')
                    st.write('The tool was last updated on May 10, 2023 and may be updated with new data as they become available.')
                    st.caption('For questions regarding model hyperparameters and development, please contact Jin Kyu (Justin) Kim at: jjk.kim@mail.utoronto.ca')
                with col_9:
                    st.image('ROC.png')
                    st.image('PRC.png')
                    st.image('CM.png')
                    st.caption('Final model evaluation using ROC, PRC, and confusion matrix.')
            
            with about_container5:
                with col_10:
                    st.subheader ('Reference')
                with col_11:
                    st.write('Predicting the likelihood of obstruction for non-refluxing primary megaureter using a calibrated ridge regression model: SK-POMP (SickKids-Primary Obstructive Megaureter Prediction)')
                    st.caption('Kim JK, Chua ME, Khondker A, ... Richter J, Lorenzo AJ, Rickard M')
                    st.caption('Pending peer-reviewed publication')

if __name__ == '__main__':
    main()
