# IntoToAIFinalProject --> GPA Prediction App with AI Generated Analysis and Advice


## Project Overview
The GPA Prediction App is a sophisticated web application developed using Streamlit, designed to predict a student's GPA based on a multitude of factors. These factors encompass personal attributes, academic involvement, and lifestyle choices, offering a comprehensive analysis of the student's profile.

This application leverages machine learning techniques to provide accurate GPA predictions, helping students understand the impact of various elements on their academic performance. Beyond just prediction, the app integrates with the Cohere API to offer personalized advice, guiding students on how to enhance their academic outcomes based on their unique profiles.

### Core Components
1. **Data Collection:** The app collects input from users regarding their previous GPA, current GPA, and various personal, academic, and lifestyle factors. This data forms the basis for the GPA prediction.

2. **Machine Learning Model:** The application uses a pre-trained machine learning model to predict the student's GPA. The model is trained on historical data and can accurately forecast GPA based on the provided inputs.

3. **Personalized Advice:** Utilizing the Cohere API, the app generates tailored advice for students. This advice is based on the predicted GPA and the individual factors influencing the student's academic performance.

4. **User-Friendly Interface:** The app features a clean and intuitive interface powered by Streamlit, making it easy for users to input their data and receive predictions and advice.


## Features
- **Comprehensive GPA Prediction:** Predicts GPA based on a combination of previous GPA, current GPA, and various factors such as motivation, nutrition, bullying experiences, sports participation, tutoring, class participation, professor quality, physical activity, parental education level, and extracurricular activities.

- **Detailed Input Form:** The app provides a structured input form where users can easily enter their information. Each input field is accompanied by helpful tooltips to guide the user.

- **Categorical Feature Encoding:** Efficiently encodes categorical variables to ensure accurate predictions by the machine learning model.

- **Integration with Cohere API:** Generates personalized advice by analyzing the student's profile and predicted GPA. The advice includes specific recommendations to help the student improve their academic performance.

- **Dynamic Advice Generation:** Depending on the predicted GPA, the advice is tailored to highlight strengths and suggest areas for improvement. For high-performing students, the app offers congratulatory messages and optional improvement suggestions, while for others, it identifies key hurdles and provides targeted advice.

- **Interactive and Responsive Design:** The Streamlit-based interface ensures that the app is interactive and responsive, providing a seamless user experience on both desktop and mobile devices.

- **Secure and Scalable:** Designed with security and scalability in mind, making it suitable for deployment in various environments, from local servers to cloud platforms.


## How to Run the Application

### Local Server
1. **Clone the Repository:**
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create a Virtual Environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Streamlit App:**
    ```sh
    streamlit run McPeggs_Final_Project_Streamlit_App.py
    ```

### Cloud Deployment
You can deploy the application to a cloud platform like Streamlit Cloud or Heroku.

#### Streamlit Cloud
1. **Create a new app on [Streamlit Cloud](https://streamlit.io/cloud).**

2. **Connect your GitHub repository.**

3. **Set the entry point to `McPeggs_Final_Project_Streamlit_App.py`.**

4. **Deploy the app.**

#### Heroku
1. **Create a `Procfile` with the following content:**
    ```sh
    web: streamlit run McPeggs_Final_Project_Streamlit_App.py
    ```

2. **Log in to Heroku and create a new app:**
    ```sh
    heroku login
    heroku create <your-app-name>
    ```

3. **Push the code to Heroku:**
    ```sh
    git add .
    git commit -m "Initial commit"
    git push heroku main
    ```

## How the Application Works

A YouTube video demonstrating how the application works can be found [here](https://youtu.be/bE2KdDgkAvY).

## Link to the Streamlit Application
The link to the streamlit app can be found [here](https://mcpeggs-gpa-prediction-app-with-ai-generated-analysis.streamlit.app/).
