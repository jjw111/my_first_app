#### Import packages
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from PIL import Image

st.title('Using Machine Learning to Drive the Future ')
st.caption('Coded by: Joshua Winter for MSBA OPIM-607 ')

image= Image.open('linkedin_photo.jpg')

st.image(image, caption='Image Sourced from Google')




st.title('Are you a LinkedIn User?')
###INCOME#####
incom = st.selectbox("Gross Household Income level", 
             options = ["Less than $10,000",
                        " 10k to $20,000",
                        " 20k to $30,000",
                        " 30k to $40,000",
                        " 40k to $50,000",
                        " 50k to $75,000",
                        " 75k to $100,000",
                        " 100k to $150,000",
                        " $150k or more?"
                         ])
st.write(f"Income selected: {incom}")

#st.write("**Convert Selection to Numeric Value**")

if incom == "Less than $10,000":
   incom = 1
elif incom == "10,000 to under $20,000":
    incom = 2
elif incom == "20,000 to under $30,000":
     incom = 3
elif incom == "30,000 to under $40,000":
    incom = 4
elif incom == "40,000 to under $50,000":
    incom = 5
elif incom == "50,000 to under $75,000":
    incom = 6
elif incom == "75,000 to under $100,000":
    incom = 7
elif incom == "100,000 to under $150,000":
    incom= 8
else:
    incom = 9
#st.write(f"Income (post-conversion): {incom}")
###INCOME#####


###EDUCATION####
educ = st.selectbox("Education level", 
             options = ["Less than High School",
                        "High School, Incomplete",
                        "High School, Graduate",
                        "Some College, No Degree",
                        "Two-Year Associate's Degree",
                        "Four-Year Bachelor's Degree",
                        "Some Postgraduate or Professional Schooling, No Degree",
                        "PostGraduate or Professional Degree"
                         ])
st.write(f"Education selected: {educ}")

#st.write("**Convert Selection to Numeric Value**")

if educ == "Less than High School":
   educ = 1
elif educ == "High School, Incomplete":
    educ = 2
elif educ == "High School, Graduate":
     educ = 3
elif educ == "Some College, No Degree":
    educ = 4
elif educ == "Two-Year Associate's Degree":
    educ = 5
elif educ == "Four-Year Bachelor's Degree":
    educ = 6
elif educ == "Some Postgraduate or Professional Schooling, No Degree":
    educ = 7
else:
    educ= 8
#st.write(f"Education (post-conversion): {educ}")
###EDUCATION####

##parent##
kid = st.selectbox("Parental Status", 
             options = ["Yes",
                        "No",
                         ])
st.write(f"Parental Status selected: {kid}")

#st.write("**Convert Selection to Numeric Value**")

if kid == "Yes":
   kid = 1
else:
    kid = 0
#st.write(f"Parental Status (post-conversion): {kid}")
##parent##

##Married##
ring = st.selectbox("Marital Status", 
             options = ["Yes",
                        "No",
                         ])
st.write(f"Marital Status selected: {ring}")

#st.write("**Convert Selection to Numeric Value**")

if ring == "Yes":
   ring = 1
else:
    ring = 0
#st.write(f"Marital Status (post-conversion): {ring}")
##Married##

##gender##
gend = st.selectbox("Gender", 
             options = ["Male",
                        "Female",
                         ])
st.write(f"Gender selected: {gend}")

#st.write("**Convert Selection to Numeric Value**")

if gend == "Female":
   gend = 1
else:
    gend = 0
#st.write(f"Gender (post-conversion): {gend}")
##gender##


#age#
age= st.number_input('Enter Your Age',
            min_value=1,
            max_value=99,
            value=30)


st.write("Your Age is: ", age)
#age#





##background data##
s = pd.read_csv(r"social_media_usage.csv")

def clean_sm(x):
    
    print(np.where(x == 1, 1, 0))
    


ss = pd.DataFrame({
    
    "sm_li":np.where(s["web1h"] == 1, 1, 0),
    
    "income":np.where(s["income"] > 9,np.nan,s["income"]),
    
    "education":np.where(s["educ2"]> 8,np.nan,s["educ2"]),
    
    "parent":np.where(s["par"] == 1,1,0),
    
    "married": np.where(s["marital"] ==1,1,0),
    
    "female": np.where(s["gender"] ==2,1,0),
    
    "age":np.where(s["age"] >98, np.nan,s["age"])})

ss = ss.dropna()

y = ss["sm_li"]

x = ss[["income", "education", "parent", "married", "female", "age"]]


x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,      # same number of target in training & test set
                                                  
                                                    test_size=0.2, # hold out 20% of data for testing
                                                    
                                                    random_state=153)  # set for reproducibility


lr = LogisticRegression(class_weight='balanced')

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

persons = pd.DataFrame({
            
    "income": [incom],
    
    "education":[educ],
    
    "parent":[kid],
    
    "married": [ring],
    
    "female": [gend],
    
    "age":[age]
})

probs = (lr.predict_proba(persons))[0][1]

probs =(round(probs,1))


st.markdown(f"Probability of being a LinkedIn User: **{probs*100 }%**")

if probs >= .8:

    isit = "Highly Likely"

elif probs > .7:

    isit = "Very Likely"

elif probs > .5:

    isit = "Likely"

else:
    
    isit = "Unlikely"

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = probs,
    title = {'text': f"LinkedIn User? {isit}"},
    gauge = {"axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, .49], "color":"white"},
                {"range": [ .5, 1], "color":"lightblue"}
            ],
            "bar":{"color":"black"}}
))

st.plotly_chart(fig)
