# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:54:05 2024

@author: Wasantha Kumara
"""

import sklearn
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import keras
from keras.models import save_model
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Loading the saved models

diabetes_model = pickle.load(open('Diabetes_Pipeline.sav','rb'))
heart_disease_model = pickle.load(open('Heart_Disease_Pipeline.sav','rb'))
parkinsons_model = pickle.load(open('Parkinsons_Pipeline .sav','rb'))
breast_cancer_model = pickle.load(open('Breast_Cancer_Pipeline.sav','rb'))

# Sidebar for navigate

with st.sidebar:

    selected = option_menu('Multiple Disease Prediction System using ML',

                           ["Diabetes Prediction",
                           "Heart  Disease Prediction",
                           "Parkinsons Prediction",
                           "Breast Cancer Predictions"],

                           icons = ['activity','heart-pulse-fill','person','person-standing-dress'],
                           default_index = 0) # This is the default starting page comes under option menu


# 1. Diabetes Prediction Page

if (selected == 'Diabetes Prediction'):

    # Page title
    st.title('Diabetes Prediction Using ML')

    # getting the input data from the user
    # columns for input fields

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input("Glucose Level")

    with col3:
        BloodPressure = st.text_input('BloodPressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value ')

    with col2:
        Insulin = st.text_input('Insulin value')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the person')

    # Pregnancies = st.text_input('Number of Pregnancies')
    # Glucose = st.text_input("Glucose Level"  )
    # BloodPressure = st.text_input('BloodPressure value')
    # SkinThickness = st.text_input('Skin Thickness value ')
    # Insulin = st.text_input('Insulin value')
    # BMI = st.text_input('BMI value')
    # DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    # Age = st.text_input('Age of the person')

    # code for prediction
    diab_diagnosis = ''

    # creating a button for prediction

    if st.button('Diabetes Test Result'):
        
        input_data = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
        diab_prediction = diabetes_model.predict(np.asarray(input_data).reshape(1, -1))
        
        
        if (diab_prediction[0]==1):
            diab_diagnosis = 'The person is Diabetic'

        else:
            diab_diagnosis = 'The person is not Diabetic'

    st.success(diab_diagnosis)
    
    
#2.  Heart  Disease Prediction

if (selected == 'Heart  Disease Prediction'):

    # Page title
    st.title('Heart  Disease Prediction Using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
       trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs =st.text_input ('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')



    # code for prediction
    Heart_diagnosis = ''

    # creating a button for prediction

    if st.button('Heart Test Result'):

        #Heart_prediction = heart_disease_model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        #user_input = [int(age), sex, int(cp), int(trestbps), int(chol), fbs, restecg, int(thalach), exang, float(oldpeak), slope, ca, thal]
        input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
        Heart_prediction = heart_disease_model.predict(np.asarray(input_data).reshape(1, -1))


        if (Heart_prediction[0]==1):
            Heart_diagnosis = 'The person is a heart patient'

        else:
            Heart_diagnosis = 'The person is not a heart patient'

    st.success(Heart_diagnosis)
    
# 3. Parkinsons  Disease Prediction

if (selected == 'Parkinsons Prediction'):

    # Page title
    st.title('Parkinsons Prediction Using ML')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')



    # code for prediction
    parkinsons_diagnosis = ''

    # creating a button for prediction

    if st.button('parkinsons Test Result'):
        #parkinsons_prediction = parkinsons_model.predict([[MDVPFo(Hz),MDVPFhi(Hz),MDVPFlo(Hz),MDVPJitter,MDVPJitter(Abs),MDVPRAP,MDVPPPQ,JitterDDP,MDVPShimmer,MDVPShimmer(dB),ShimmerAPQ3,ShimmerAPQ5,MDVPAPQ,ShimmerDDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
        input_data = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        
        parkinsons_prediction = parkinsons_model.predict(np.asarray(input_data).reshape(1, -1))


        if (parkinsons_prediction[0]==1):
            parkinsons_diagnosis = "The person has Parkinson's"

        else:
            parkinsons_diagnosis = "The person does not have Parkinson's Disease"

    st.success(parkinsons_diagnosis)


# 4. Breast Cancer Predictions

if (selected == 'Breast Cancer Predictions'):

    # Page title
    st.title('Breast Cancer Predictions Using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        mean_radius = st.text_input('mean radius')

    with col2:
        mean_texture = st.text_input('mean texture')

    with col3:
        mean_perimeter = st.text_input('mean perimeter')

    with col1:
        mean_area = st.text_input('mean area')

    with col2:
        mean_smoothness = st.text_input('mean smoothness')

    with col3:
        mean_compactness = st.text_input('mean compactness')

    with col1:
        mean_concavity =st.text_input ('mean concavity')

    with col2:
        mean_concave_points = st.text_input('mean concave points')

    with col3:
        mean_symmetry = st.text_input('mean symmetry')

    with col1:
        mean_fractal_dimension = st.text_input('mean fractal dimension')

    with col2:
        radius_error= st.text_input('radius error')

    with col3:
        texture_error = st.text_input('texture error')

    with col1:
        perimeter_error = st.text_input('perimeter error')

    with col2:
        area_error = st.text_input('area error')

    with col3:
        smoothness_error= st.text_input('smoothness error')

    with col1:
        compactness_error = st.text_input('compactness error')

    with col2:
        concavity_error = st.text_input('concavity error')

    with col3:
        concave_points_error = st.text_input("concave points error")

    with col1:
        symmetry_error =st.text_input ('symmetry error')

    with col2:
        fractal_dimension_error = st.text_input('fractal dimension error')

    with col3:
        worst_radius = st.text_input('worst radius')

    with col1:
        worst_texture = st.text_input('worst texture')

    with col2:
        worst_perimeter = st.text_input('worst perimeter')

    with col3:
        worst_area = st.text_input('worst area')

    with col1:
        worst_smoothness = st.text_input('worst smoothness')

    with col2:
        worst_compactness = st.text_input('worst compactness')

    with col3:
        worst_concavity = st.text_input("worst concavity")

    with col1:
        worst_concave_points =st.text_input ('worst concave points')

    with col2:
        worst_symmetry = st.text_input('worst symmetry')

    with col3:
        worst_fractal_dimension = st.text_input('worst fractal dimension')



    # code for prediction
    breast_cancer_diagnosis = ''

    # creating a button for prediction

    if st.button('Breast Cancer Prediction Test Result'):

        #breabreast_cancer_prediction =breast_cancer_model.predict([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]])
        input_data = [float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area),
                      float(mean_smoothness), float(mean_compactness), float(mean_concavity), float(mean_concave_points),
                      float(mean_symmetry), float(mean_fractal_dimension), float(radius_error), float(texture_error),
                      float(perimeter_error), float(area_error), float(smoothness_error), float(compactness_error),
                      float(concavity_error), float(concave_points_error), float(symmetry_error), float(fractal_dimension_error),
                      float(worst_radius), float(worst_texture), float(worst_perimeter), float(worst_area),
                      float(worst_smoothness), float(worst_compactness), float(worst_concavity), float(worst_concave_points),
                      float(worst_symmetry), float(worst_fractal_dimension)]

        # Perform prediction
        breast_cancer_prediction = breast_cancer_model.predict(np.asarray(input_data).reshape(1, -1))
        #prediction_label = [np.argmax(breast_cancer_prediction)]

        if (breast_cancer_prediction[0]==0):

            breast_cancer_diagnosis = 'The person does have a Brest Cancer'

        else:
            breast_cancer_diagnosis = 'The person does not have a breast cancer'

    st.success(breast_cancer_diagnosis)
