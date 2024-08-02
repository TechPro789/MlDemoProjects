import os
import pickle
import re
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes


weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"] ## ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stopsign',]
img_preprocess = weights.transforms() ## Scales values from 0-255 range to 0-1 range.




# Set page configuration
#st.set_page_config(page_title="Health Assistant",##
 #                  layout="wide",
  #                 page_icon="🧑‍⚕️")
  # Set page title and favicon
st.set_page_config(page_title="Zeeshan Shaikh's ML Portfolio", page_icon="🧑‍⚕️")
 
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

sentiment_model  = pickle.load(open(f'{working_dir}/saved_models/sentiment_analysis.pkl', 'rb'))

LoanPredict_model = pickle.load(open(f'{working_dir}/saved_models/ML_Model.pkl', 'rb'))

Fake_model = pickle.load(open(f'{working_dir}/saved_models/fake_news_model.pkl', 'rb'))

vector_form = pickle.load(open(f'{working_dir}/saved_models/vector_model.pkl', 'rb'))

# Fake_model =  pickle.load(open(f'{working_dir}/saved_models/model.pkl', 'rb'))
#For Object Detection
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval(); ## Setting Model for Evaluation/Prediction   
    return model

model = load_model()


def make_prediction(img): 
    img_processed = img_preprocess(img) ## (3,500,500) 
    prediction = model(img_processed.unsqueeze(0)) # (1,3,500,500)
    prediction = prediction[0]                       ## Dictionary with keys "boxes", "labels", "scores".
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction): ## Adds Bounding Boxes around original Image.
    img_tensor = torch.tensor(img) ## Transpose
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="person" else "green" for label in prediction["labels"]] , width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) ### (3,W,H) -> (W,H,3), Channel first to channel last.
    return img_with_bboxes_np

##End

st.markdown(
    """
    <style>
    .title-font {
        font-size: 40px;
        color: #4CAF50; /* Optional: Change the color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# sidebar for navigation
with st.sidebar:
    selected = option_menu('Zeeshan Shaikh',

                           ['Home',
                            'Loan Prediction System',
                            'Fake News Classifier','Object Detection','Sentiment Analysis','Diabetes Prediction'],
                           menu_icon='hospital-fill',
                           icons=[],
                           default_index=0)


if selected == "Home":


    # Custom CSS for better styling
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        .medium-font {
            font-size:24px !important;
            font-weight: bold;
        }
        .project-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="big-font">Welcome to My Machine Learning Portfolio</p>', unsafe_allow_html=True)

    # Introduction
    st.write("""
    Hello! I'm Zeeshan Shaikh, a passionate machine learning enthusiast. This portfolio showcases the projects 
    I've developed during my machine learning certification journey with Simplilearn. Each project demonstrates my skills 
    in applying various ML techniques to real-world problems.
    """)

    # Skills section
    st.markdown('<p class="medium-font">Skills</p>', unsafe_allow_html=True)
    st.write("Python | Pandas | Scikit-learn | OpenCv | NumPy | Data Visualization")

    # Projects section
    st.markdown('<p class="medium-font">Featured Projects</p>', unsafe_allow_html=True)

    # Project 1
    with st.container():
        st.subheader("1. Loan Prediction System.")
        st.write("""
        Developed a machine learning model using a dataset from Kaggle to predict loan eligibility 
        with Support Vector Machines (SVM), automating and improving the loan approval process by
        determining whether a loan applicant is eligible.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Project 2
    with st.container():
        st.subheader("2. Fake News Classifier.")
        st.write("""
        Built a Fake News Detection System trained on FakeNewsNet training and testing data using
        the Random Forest algorithm to identify and classify news articles as either fake or real.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Project 3
    with st.container():
        st.subheader("3. Object Detection.")
        st.write("""
        Built an object detection model using computer vision and pre-trained deep learning models 
        to identify and locate objects within a given image.

        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.subheader("4. Sentiment Analysis.")
        st.write("""
        Built a NLP model to analyze sentiment in movie reviews. Used NLTK for text preprocessing 
        and an LSTM neural network, achieving 88% accuracy on the IMDB dataset."
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.subheader("5. Diabetes Prediction.")
        st.write("""
        Developed an automated system for early prediction and diagnosis of diabetes using the
        Decision Tree algorithm. I used the PIMA dataset to train the model."  
        
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    # Contact section
    st.markdown('<p class="medium-font">Get in Touch</p>', unsafe_allow_html=True)
    st.write("Email: zeeshan.shaikh.shahid@gmail.com")
    st.write("LinkedIn: [Zeeshan Shaikh](https://www.linkedin.com/in/zeeshan-shaikh-9b1856264?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")
    st.write("Tableau Public: [ZeeshanShaikh_TBL](https://public.tableau.com/app/profile/zeeshan.shaikh3147/vizzes)")
    st.write("Contact/WatsApp: (+852 5421 1979)")
    

 
# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])
        
        st.write("### Model Details:")
        st.write(diabetes_model)

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

if selected == "Sentiment Analysis":
    
# create title
    st.title('Sentiment Analysis Model')

    review = st.text_input('Enter your review:')

    submit = st.button('Predict')

    if submit:
        prediction = sentiment_model.predict([review])

        if prediction[0] == 'positive':
            st.success('Positive Review')
        else:
            st.warning('Negative Review')

if selected == "Fake News Classifier":
    st.title('Fake News Classifier')
    input_text = st.text_input('Enter news Article')

    def prediction(input_text):
        input_data = vector_form.transform([input_text])
        prediction = Fake_model.predict(input_data)
        return prediction[0]

    if input_text:
        pred = prediction(input_text)
        if pred == 1:
            st.warning('The News is Fake')
        else:
            st.success('The News Is Real')
         
    

if selected == "Loan Prediction System":
     st.title("Bank Loan Prediction using Machine Learning")

    ## Account No
     account_no = st.text_input('Account number')

    ## Full Name
     fn = st.text_input('Full Name')

    ## For gender
     gen_display = ('Female','Male')
     gen_options = list(range(len(gen_display)))
     gen = st.selectbox("Gender",gen_options, format_func=lambda x: gen_display[x])

    ## For Marital Status
     mar_display = ('No','Yes')
     mar_options = list(range(len(mar_display)))
     mar = st.selectbox("Marital Status", mar_options, format_func=lambda x: mar_display[x])

    ## No of dependets
     dep_display = ('No','One','Two','More than Two')
     dep_options = list(range(len(dep_display)))
     dep = st.selectbox("Dependents",  dep_options, format_func=lambda x: dep_display[x])

    ## For edu
     edu_display = ('Not Graduate','Graduate')
     edu_options = list(range(len(edu_display)))
     edu = st.selectbox("Education",edu_options, format_func=lambda x: edu_display[x])

    ## For emp status
     emp_display = ('Job','Business')
     emp_options = list(range(len(emp_display)))
     emp = st.selectbox("Employment Status",emp_options, format_func=lambda x: emp_display[x])

    ## For Property status
     prop_display = ('Rural','Semi-Urban','Urban')
     prop_options = list(range(len(prop_display)))
     prop = st.selectbox("Property Area",prop_options, format_func=lambda x: prop_display[x])

    ## For Credit Score
     cred_display = ('Between 300 to 500','Above 500')
     cred_options = list(range(len(cred_display)))
     cred = st.selectbox("Credit Score",cred_options, format_func=lambda x: cred_display[x])

    ## Applicant Monthly Income
     mon_income = st.number_input("Applicant's Monthly Income($)",value=0)

    ## Co-Applicant Monthly Income
     co_mon_income = st.number_input("Co-Applicant's Monthly Income($)",value=0)

    ## Loan AMount
     loan_amt = st.number_input("Loan Amount",value=0)

    ## loan duration
     dur_display = ['2 Month','6 Month','8 Month','1 Year','16 Month']
     dur_options = range(len(dur_display))
     dur = st.selectbox("Loan Duration",dur_options, format_func=lambda x: dur_display[x])
     predict_elig = st.button("Predict Eligibility")
     duration = 0
     if dur == 0:
            duration = 60
     if dur == 1:
            duration = 180
     if dur == 2:
            duration = 240
     if dur == 3:
            duration = 360
     if dur == 4:
            duration = 480
     if predict_elig:

        features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
        print(features)
        prediction = LoanPredict_model.predict(features)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))
        if ans == 0:
            st.error(
                "Hello: " + fn +" || "
                "Account number: "+account_no +' || '
                'According to our Calculations, you will not get the loan from Bank'
            )
        else:
            st.success(
                "Hello: " + fn +" || "
                "Account number: "+account_no +' || '
                'Congratulations!! you will get the loan from Bank'
            )
if selected == "Object Detection":
    st.title("Object Detector :tea: :coffee:")
    upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

    if upload:

        img = Image.open(upload)

        prediction = make_prediction(img) ## Dictionary
        img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction) ## (W,H,3) -> (3,W,H)

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        plt.imshow(img_with_bbox)
        plt.xticks([],[])
        plt.yticks([],[])
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

        st.pyplot(fig, use_container_width=True)

        del prediction["boxes"]
        st.header("Predicted Probabilities")
        st.write(prediction)
