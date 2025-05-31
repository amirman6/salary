# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:19:05 2022

@author: T430s
"""
import streamlit as st
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from streamlit_option_menu import option_menu

st.write("""
#### Salary Prediction and Analysis App
--by A. Maharjan 
""")
image = Image.open('salary.jpg')
st.image(image)


selected = option_menu(
    menu_title= None,
    options= ["Salary Prediction","Analytics"],
    menu_icon = "cast",
    #default_index = 0,
    orientation= "horizontal",
    styles = {
        "nav-link":{"font-size":"14px","text-align":"center"},
        "nav-link-selected": {"background-color":"green"}
        }
    
    )


    
   
if selected == "Salary Prediction":
    st.write('(Choose the variables from the left side bar)')
    # loading the saved model for Home Price(random forest model was used)
    df_salary = pickle.load(open('df_salary','rb'))

    #st.write('---')

    # Sidebar
    # Header of Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    #def user_input_features():
    Gender = np.append(' ',df_salary['Gender'].unique())
    Gender = st.sidebar.selectbox('Gender', Gender)
    
    EducationLevel = np.append(' ',df_salary['Education Level'].unique())
    EducationLevel = st.sidebar.selectbox('Education Level', EducationLevel)
    
    JobTitle = np.append(' ', df_salary['Job Title'].sort_values().unique())
    JobTitle = st.sidebar.selectbox('Job Title', JobTitle)
    
    
    #Age = st.sidebar.slider('Age',df_salary['Age'].min(),df_salary['Age'].max(),int(df_salary['Age'].mean()),1)
    
    Experience = st.sidebar.slider("Years of Experience",df_salary["Years of Experience"].min(),df_salary["Years of Experience"].max(),df_salary["Years of Experience"].mean(),1.0)
    

    # dataframe for model prediction
    X_new = pd.DataFrame([[Gender,EducationLevel,JobTitle,Experience]],
                         columns=['Gender', 'Education Level', 'Job Title', 'Years of Experience'])



    # Main Panel
    # Apply Model to Make Prediction
    
    # results output
    if st.button('Predict Salary'):
        n_bootstraps = 50
        model_list=[]
        for i in range(n_bootstraps):
            model_list.append('model'+str(i))  

        # run below codes to predict
        #path="model_folder/" # for xgboost make sure to put all the model files into this
        path="model_folder_stacking/" # for stacking
        
        all_preds = []
        for i in model_list:
            model_stacking = pickle.load(open(path + i,'rb'))
            preds = model_stacking.predict(X_new)
            all_preds.append(preds)
            

        # Convert to NumPy array: shape (n_bootstraps, n_samples)
        all_preds = np.array(all_preds)
        

        # Compute statistics
        mean_preds = np.mean(all_preds, axis=0)
        lower_ci = np.percentile(all_preds, 2.5, axis=0)
        upper_ci = np.percentile(all_preds, 97.5, axis=0)
        st.write('- Estimated Predicted Salary is: $ ','  ','%.2f' %mean_preds[0])
        st.write('- With 95% Lower Confidence Interval: $','  ','%.2f' %lower_ci[0])
        st.write('- With 95% Upper Confidence Interval: $','  ','%.2f' %upper_ci[0])
        
        fig,ax = plt.subplots()
        plt.boxplot(all_preds)
        #plt.show() # does not really need this for st pyplot
        plt.title('Predicted Salary Range in Box (percentile) plot')
        plt.ylabel('Salar in US $')
        st.pyplot(fig)


    

     
       


# analytics
if selected == "Analytics":
    import seaborn as sns
    df_salary = pickle.load(open('df_salary','rb'))

    st.write("""
    #### Histograms
    """)
    
    out = st.selectbox('select',(' ','Salary','Years of Experience'))
    option = st.selectbox('choose to compare',(' ','Gender','Education Level'))
    
    if st.button('Press for the histogram'):
        fig,ax = plt.subplots()
        sns.histplot(data = df_salary,x=out,hue=option)
        #plt.show() # does not really need this for st pyplot
        plt.title('Histogram of Salary')
        st.pyplot(fig)
    st.write("-----")





    # scatterplots
    st.write("""
    #### Check Relationships
    """)

    X = st.selectbox('X',(' ','Years of Experience'))
    Y = st.selectbox('Y',(' ','Salary'))
    options = st.selectbox('choose to compare with',(' ','Gender','Education Level'))
    if st.button('Press to see the relationship'):
        fig,ax = plt.subplots()
        sns.scatterplot(data = df_salary,x = X,y = Y,hue = options)
        
        st.pyplot(fig)



    # barplots
    st.write("""
    #### Barplots
    """)
    #option = st.selectbox('choose to compare',('Origin','DriveTrain'))
    out_barplot = st.selectbox('Choose for the barplot',(' ','Salary'))
    option2 = st.selectbox('choose to compare with',(' ','Gender','Education Level',''))
    if st.button('Press for the BarPlot'):
        fig,ax = plt.subplots()
        sns.barplot(df_salary, x = option2, y = out_barplot) # y is the nemerical value of the mean for the bar plot
        plt.ylabel('Salary')
        plt.xticks(rotation=90) # makes the x axis names vertical
        plt.tight_layout() # expands more horizotally
       
        st.pyplot(fig)

    st.write("-----")



    # pairplot
    st.write("""
    #### PairPlot. May take few seconds
    #""")
    
    #option3 = st.selectbox('compare for pairplot',('','Gender','Education Level'))  
    
    if st.button('Press for PairPlot'):
        
        df_salary_pairplot = df_salary[['Salary','Years of Experience']]

        fig,ax = plt.subplots()

        pairplot = sns.pairplot(data = df_salary_pairplot,diag_kind='hist', height=3, aspect=1.2)
        plt.show()
        st.pyplot(pairplot.fig) # pairplot needs this syntax to work, got from chatgpt
        
        







