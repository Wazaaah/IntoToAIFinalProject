import streamlit as st
import pandas as pd
import requests
import json
import joblib

# Load the model
model = joblib.load('gpa_prediction_model.pkl')


# Define a function to encode the categorical variables
def encode_categorical_features(data):
    encoding_map = {
        'Previous_Grades': {'A': 0, 'B': 1, 'C': 2},
        'Grades': {'A': 0, 'B': 1, 'C': 2},
        'Motivation': {'Low': 1, 'Medium': 2, 'High': 0},
        'Nutrition': {'Unhealthy': 2, 'Healthy': 1, 'Balanced': 0},
        'Bullying': {'No': 0, 'Yes': 1},
        'Sports_Participation': {'Low': 1, 'Medium': 2, 'High': 0},
        'Tutoring': {'No': 0, 'Yes': 1},
        'Class_Participation': {'Low': 1, 'Medium': 2, 'High': 0},
        'Professor_Quality': {'Low': 1, 'Medium': 2, 'High': 0},
        'Physical_Activity': {'Low': 1, 'Medium': 2, 'High': 0},
        'Parental_Education': {'High School': 2, 'Some College': 3, 'College': 0, 'Graduate': 1},
        'Extracurricular_Activities': {'No': 0, 'Yes': 1},
        'Gender': {'Male': 1, 'Female': 0},
        'Educational_Tech_Use': {'Yes': 1, 'No': 0},
        'Stress_Levels': {'Low': 1, 'Medium': 2, 'High': 0},
        'Study_Space': {'No': 0, 'Yes': 1},
        'Mentoring': {'No': 0, 'Yes': 1},

    }
    return data.replace(encoding_map)


# Define a function to get advice from the Cohere API
def get_advice_from_cohere(user_data, highest_possible_gpa):
    cohere_api_key = "omFu9KFgafnalHdiwZX7qxiAOaiK7sX8HQo5XzCA"  # Secure your API key
    cohere_api_url = "https://api.cohere.ai/v1/generate"

    headers = {
        "Authorization": f"Bearer {cohere_api_key}",
        "Content-Type": "application/json"
    }

    prompt = (f"The following data represents a student's profile and Predicted GPA along with the Highest "
              f"possible GPA:\n\n"
              f"Previous GPA: {user_data['previous_gpa']}\n"
              f"Previous Grade For Last Semester: {user_data['Previous_Grades']}\n"
              f"Current Grade For The Semester: {user_data['Grades']}\n"
              f"Motivation Level: {user_data['Motivation']}\n"
              f"Nutrition Level: {user_data['Nutrition']}\n"
              f"Bullying Experienced: {user_data['Bullying']}\n"
              f"Sports Participation: {user_data['Sports_Participation']}\n"
              f"Received Tutoring: {user_data['Tutoring']}\n"
              f"Class Participation: {user_data['Class_Participation']}\n"
              f"Professor Quality: {user_data['Professor_Quality']}\n"
              f"Physical Activity: {user_data['Physical_Activity']}\n"
              f"Parental Education Level: {user_data['Parental_Education']}\n"
              f"Extracurricular Activities: {user_data['Extracurricular_Activities']}\n"
              f"Family Income: {user_data['Family_Income']}\n"
              f"Gender: {user_data['Gender']}\n"
              f"Educational Technology Usage: {user_data['Educational_Tech_Use']}\n"
              f"Stress Levels: {user_data['Stress_Levels']}\n"
              f"Study Space: {user_data['Study_Space']}\n"
              f"Mentoring: {user_data['Mentoring']}\n"
              f"Class Size: {user_data['Class_Size']}\n"
              f"Predicted GPA: {user_data['Predicted GPA']}\n\n"
              f"Highest Possible GPA: {highest_possible_gpa}\n\n"
              "Please analyze the student's profile and provide advice based only on the Predicted GPA.\n"
              "Use the following guidelines:\n"
              "The Highest Possible GPA is a 4.0\n"
              "1. If the Predicted GPA is between 3.85 and 4.0:\n"
              "   - Start by congratulating the student on their excellent performance.\n"
              "   - Provide optional suggestions for further improvement without mentioning the GPA range.\n"
              "2. If the Predicted GPA is below 3.85:\n"
              "   - Identify the major hurdles affecting the student's GPA based on the provided data.\n"
              "   - Give targeted advice to help the student improve their GPA.\n\n"
              "Begin directly with the analysis or advice. Do not include any introductory statements, explanations, "
              "or apologies."
              "Always remember to give at least 4 advices and at most 7. The length of each advice should not be less "
              "than 5 lines."
              "Also, you are speaking directly to the student, so make it sound as such."
              )

    payload = {
        "model": "command-xlarge",
        "prompt": prompt,
        "max_tokens": 1000  # Adjust as needed
    }

    response = requests.post(cohere_api_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result['generations'][0]['text'].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"


# Define a function to get graduation honors based on GPA
def get_graduation_honors(gpa):
    if gpa >= 3.85:
        return "Summa Cum Laude (Highest Honors) 🎓🌟🏅"
    elif gpa >= 3.70:
        return "Magna Cum Laude (High Honors) 🎓🥈"
    elif gpa >= 3.50:
        return "Cum Laude (Honors) 🥉"
    else:
        return "None 😔"


# Define the Streamlit app
st.set_page_config(
    page_title="GPA Prediction App",
    page_icon="🎓",
    layout="centered"
)
# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'landing'


# Landing Page Function
def landing_page():
    st.markdown("""
    <style>
    .falling-emojis {
        position: fixed; /* Ensure emojis are fixed to the viewport */
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none; /* Allow clicks through emojis */
        overflow: hidden; /* Prevent scrollbars */
        z-index: 1000; /* Ensure emojis are on top */
    }
    .emoji {
        position: absolute;
        font-size: 2rem;
        animation: fall linear infinite;
    }
    @keyframes fall {
        0% {
            transform: translateY(-100px);
            opacity: 1;
        }
        100% {
            transform: translateY(calc(100vh + 100px));
            opacity: 0;
        }
    }
    .emoji:nth-child(1) {
        left: 10%;
        animation-duration: 5s; /* Faster speed */
        animation-delay: 0s;
    }
    .emoji:nth-child(2) {
        left: 30%;
        animation-duration: 6s; /* Faster speed */
        animation-delay: 1s;
    }
    .emoji:nth-child(3) {
        left: 50%;
        animation-duration: 7s; /* Faster speed */
        animation-delay: 2s;
    }
    .emoji:nth-child(4) {
        left: 70%;
        animation-duration: 5s; /* Faster speed */
        animation-delay: 3s;
    }
    .emoji:nth-child(5) {
        left: 90%;
        animation-duration: 6s; /* Faster speed */
        animation-delay: 4s;
    }
    </style>
    <div class="falling-emojis">
        <div class="emoji">🎓</div>
        <div class="emoji">✨</div>
        <div class="emoji">🌟</div>
        <div class="emoji">🎉</div>
        <div class="emoji">🎊</div>
    </div>
    """, unsafe_allow_html=True)

    st.title('Welcome to the GPA Prediction App!')
    st.markdown("""
    ### About This App
    This application helps you predict your GPA based on various personal, academic, and lifestyle factors. 

    **How it Works:** 1. **Enter Your Details:** Provide information about your grades, motivation, nutrition, 
    and other relevant factors. 2. **Get Your Predicted GPA:** Based on the data you provide, the app will predict 
    your GPA. 3. **Receive Personalized Advice:** The app will analyze your profile and offer tailored advice to help 
    you improve your GPA if needed.

    **Congratulations!** 🎉 If your predicted GPA is high, we celebrate your success. If not, we offer suggestions for improvement.
    """)

    if st.button('Start'):
        st.session_state.page = 'input_form'


st.title('GPA Prediction App')
st.markdown("""
<style>
    .stButton>button {background-color: #4CAF50; color: white; border: none; padding: 10px 24px; text-align: center; font-size: 16px; cursor: pointer;}
    .stButton>button:hover {background-color: #45a049;}
</style>
""", unsafe_allow_html=True)


# Input Form Page
def input_form():
    st.subheader('Enter your details to predict your GPA and get advice')

    st.markdown("### Personal Factors")
    col1, col2, col3 = st.columns(3)
    with col1:
        Gender = st.radio('Gender', ('Male', 'Female'), help="Select your gender")
    with col2:
        bullying = st.radio("Bullying Experienced", ('No', 'Yes'), help="Have you experienced bullying?")
    with col3:
        tutoring = st.radio("Received Tutoring", ('No', 'Yes'), help="Have you received any tutoring?")

    col1, col2, col3 = st.columns(3)
    with col1:
        motivation = st.radio("Motivation Level", ('Low', 'Medium', 'High'), help="Select your level of motivation.")
    with col2:
        sports_participation = st.radio("Sports Participation", ('Low', 'Medium', 'High'),
                                        help="Select your level of sports participation.")
    with col3:
        nutrition = st.radio("Nutrition Level", ('Unhealthy', 'Healthy', 'Balanced'),
                             help="Select your nutritional habits.")

    col1, col2, col3 = st.columns(3)
    with col1:
        Stress_Levels = st.radio("Stress Levels", ('Low', 'Medium', 'High'), help="Select your stress level.")
    with col2:
        Mentoring = st.radio("Received Mentoring", ('No', 'Yes'), help="Have you received any mentoring?")

    st.markdown("### Academic Factors")
    col1, col2 = st.columns(2)
    with col1:
        previous_gpa = st.number_input('Previous GPA', min_value=0.0, max_value=4.0, step=0.1,
                                       help="Enter your previous GPA (0.0 to 4.0).")
        Previous_Grades = st.radio("Previous Grades", ('A', 'B', 'C'), help="Select your previous Grades.")
    with col2:
        Class_Size = st.number_input('Class Size', min_value=20.0, max_value=100.0, step=0.1)
        Grades = st.radio("Current Grades", ('A', 'B', 'C'), help="Select your current Grades.")

    col1, col2 = st.columns(2)
    with col1:
        class_participation = st.radio("Class Participation", ('Low', 'Medium', 'High'),
                                       help="Select your level of class participation.")
    with col2:
        professor_quality = st.radio("Professor Quality", ('Low', 'Medium', 'High'),
                                     help="Rate the quality of your professors.")

    col1, col2 = st.columns(2)
    with col1:
        Study_Space = st.radio("Study Space", ('No', 'Yes'), help="Select whether or not you have a study space.")

    with col2:
        Educational_Tech_Use = st.radio("Educational Technology Usage", ('No', 'Yes'),
                                        help="Select whether or not you use educational technology.")

    st.markdown("### Lifestyle Factors")
    col1, col2, col3 = st.columns(3)
    with col1:
        physical_activity = st.radio("Physical Activity", ('Low', 'Medium', 'High'),
                                     help="Select your level of physical activity.")
    with col2:
        parental_education = st.radio("Parental Education Level",
                                      ('High School', 'Some College', 'College', 'Graduate'),
                                      help="Select the highest education level achieved by your parents.")
    with col3:
        extracurricular_activities = st.radio("Extracurricular Activities", ('No', 'Yes'),
                                              help="Do you participate in extracurricular activities?")

    col1, col2, col3 = st.columns(3)
    with col1:
        Family_Income = st.number_input('Family Income', min_value=0.0, max_value=7500.0, step=500.0,
                                        help="Enter your family income.")

    submit_button = st.form_submit_button(label='Predict GPA')

    # Prediction and Display
    if submit_button:
        input_data = pd.DataFrame({
            'previous_gpa': [previous_gpa],
            'Grades': [Grades],
            'Previous_Grades': [Previous_Grades],
            'Motivation': [motivation],
            'Nutrition': [nutrition],
            'Bullying': [bullying],
            'Sports_Participation': [sports_participation],
            'Tutoring': [tutoring],
            'Class_Participation': [class_participation],
            'Professor_Quality': [professor_quality],
            'Physical_Activity': [physical_activity],
            'Parental_Education': [parental_education],
            'Extracurricular_Activities': [extracurricular_activities],
            'Family_Income': [Family_Income],
            'Gender': [Gender],
            'Educational_Tech_Use': [Educational_Tech_Use],
            'Stress_Levels': [Stress_Levels],
            'Study_Space': [Study_Space],
            'Mentoring': [Mentoring],
            'Class_Size': [Class_Size],
        })

        input_data_encoded = encode_categorical_features(input_data)

        prediction = model.predict(input_data_encoded)
        st.write(f'## Predicted GPA: {prediction[0]:.2f}')

        # Display graduation honors
        honors = get_graduation_honors(prediction)
        st.write(f'### Graduation Honors: {honors}')

        user_data = {
            'previous_gpa': previous_gpa,
            'Grades': Grades,
            'Previous_Grades': Previous_Grades,
            'Motivation': motivation,
            'Nutrition': nutrition,
            'Bullying': bullying,
            'Sports_Participation': sports_participation,
            'Tutoring': tutoring,
            'Class_Participation': class_participation,
            'Professor_Quality': professor_quality,
            'Physical_Activity': physical_activity,
            'Parental_Education': parental_education,
            'Extracurricular_Activities': extracurricular_activities,
            'Family_Income': Family_Income,
            'Gender': Gender,
            'Educational_Tech_Use': Educational_Tech_Use,
            'Stress_Levels': Stress_Levels,
            'Study_Space': Study_Space,
            'Mentoring': Mentoring,
            'Class_Size': Class_Size,
            'Predicted GPA': prediction[0]
        }
        highest_possible_gpa = 4

        advice = get_advice_from_cohere(user_data, highest_possible_gpa)
        st.write("### Advice and Hurdles Analysis")
        st.write(advice)


# Main app logic
if st.session_state.page == 'landing':
    landing_page()
elif st.session_state.page == 'input_form':
    with st.form(key='gpa_form'):
        input_form()
