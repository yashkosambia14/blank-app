import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return(np.where(x == 1, 1, 0))

ss = pd.DataFrame({"sm_li":clean_sm(s['web1h']),
    "income":np.where(s['income']>9,np.nan,s['income']),
    "education":np.where(s['educ2']>8,np.nan,s['educ2']),
    "parent":np.where(s['par']==1,1,0),
    "married":np.where(s['marital']==1,1,0),
    "female":np.where(s['gender']==2,1,0),
    "age":np.where(s['age']>98,np.nan,s['age'])
})

ss.dropna(inplace = True) # no more missing values!

Y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    stratify=Y,       
                                                    test_size=0.2,    
                                                    random_state=123)

lr = LogisticRegression(class_weight="balanced")

# fitting lr to training data

lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

# Making some predictions
#user1 = [8, 5, 0, 1, 1, 42]

# Predict class, given input features
#predicted_class = lr.predict([user1])

# Generate probability of positive class (=1)
#probs = lr.predict_proba([user1])

# Print predicted class and probability
#print(f"Predicted class: {predicted_class[0]}") 
#print(f"Probability that this person is on LinkedIn: {probs[0][1]}")

###########################################################################

def predict(data):
    mod = lr
    return mod.predict(data)

def proba(data):
    pro2 = lr
    return pro2.predict_proba(data)

    ##########################################################################

st.image(image = "https://www.siliconrepublic.com/wp-content/uploads/2022/10/a-3-718x523.jpeg")

st.markdown("# Welcome to David's LinkedIn user predictor application!")
st.markdown("## This application uses a classification model to predict whether or not a person is on LinkedIn, based on parameters provided by the user below.")

st.markdown("#### ____________________________________________________________")


st.markdown("### Please answer the below questions for the prediction:")

col1, col2 = st.columns(2)
with col1:
    income = st.selectbox("What annual income range does this person fall into?",["Less than $10K", "$10K-$20K", "$20K-$30K", "$30K-$40K", "$40K-$50K", "$50K-$75K", "$75K-$100K", "$100K-$150K", "At least $150K"])
    education = st.selectbox("What is the eduation level of the person?", ["Less than High School", "Enrolled in High School", "High School Graduate", "Enrolled in College", "Two-Year Degree", 
    "Four-Year Degree", "Enrolled in Postgraduate Program", "Postgraduate Degree"])
    parent = st.selectbox("Is the person a parent of a child living under 18 in their home?", ["Yes", "No"])
if income == "Less than $10K":
    income = 1
elif income == "$10K-$20K":
    income = 2
elif income == "$20K-$30K":
    income = 3
elif income == "$30K-$40K":
    income = 4
elif income == "$40K-$50K":
    income = 5
elif income == "$50K-$75K":
    income = 6
elif income == "$75K-$150K":
    income = 7
else: income = 8

if education == "Less than High School":
    education = 1
elif education == "Enrolled in High School":
    education = 2
elif education == "High School Graduate":
    education = 3
elif education == "Enrolled in College":
    education = 4
elif education == "Two-Year Degree":
    education = 5
elif education == "Four-Year Degree":
    education = 6
elif education == "Enrolled in Postgraduate Program":
    education = 7
else: education = 8

if parent == "Yes":
    parent = 1
else:
    parent = 0

with col2:
    married = st.selectbox("Is the person married?", ["Yes", "No"])
    female = st.selectbox("Is the person a male or female?", ["Male", "Female"])
    age = age = st.slider(label="What is the person's age?", min_value=1, max_value=98,step=1)

if married == "Yes":
    married = 1
else:
    married = 0

if female == "Female":
    female = 1
else:
    female = 0

data = [income, education, parent, married, female, age]

if st.button("Click to predict if this user is on LinkedIn"):
    result = predict(np.array([[income, education, parent, married, female, age]]))
    prob = proba(np.array([[income, education, parent, married, female, age]]))
    if result == 1:
        print(st.text("This person is likely on LinkedIn."))
        st.text(f"Probability that this person is a LinkedIn user: {prob[0][1]}")
    else:
        print(st.text("This person is likely NOT on LinkedIn."))
        st.text(f"Probability that this person is a LinkedIn user: {prob[0][1]}")

