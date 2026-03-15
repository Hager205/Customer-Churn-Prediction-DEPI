import json
import math
import time
import datetime
import pandas as pd
from PIL import Image
import streamlit as st
from pickle import load
from streamlit_lottie import st_lottie
import logging

model_file = 'model/best_xgb_model.pkl'
scaler_file = 'model/scaler.pkl'

# Load the model
with open(model_file, 'rb') as f_in:
    model = load(f_in)

# Load the scaler
with open(scaler_file, 'rb') as f_in:
    scaler = load(f_in)

# Set up logging
logging.basicConfig(
    filename='streamlit_app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s' # format the log message (timestamp, level of the message, message) ex. 12:00:00 INFO <message>
)
logger = logging.getLogger(__name__) # get the logger instance

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
       
lottie_file = './assets/Animation3.json'

lottie_file = load_lottiefile(lottie_file)

def main():
    image = Image.open("./assets/Label_Image.png")
    image2 = Image.open('./assets/MCAR.png')
    image3 = Image.open('./assets/proAr.png')
    st.image(image, width=900)  # reduces the image width to 300 pixels

    # Display images side by side in the sidebar
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.image(image3, use_container_width=True)
    with col2:
        st.image(image2, use_container_width=True)
    
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Single entry", "Batch entry")
    )

    # Project Team Section with stylish design
    st.sidebar.markdown("---")

    # Add a lottie animation to the sidebar
    with st.sidebar:
        st_lottie(lottie_file, speed=1, loop=True, quality="high", height=300, width=300)

    # Project Team Section with stylish design
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='text-align: center; font-size:30px;'>✨ Project Team ✨</h3>", unsafe_allow_html=True)
    
    # Project Team Members
    team_html = """
        <div>
            <ul style="list-style-type:none; font-size:16px; text-align:left; padding-left: 0;">
            <li style="margin-bottom: 12px;">
                👨‍💻 <strong>Youssef Ayman Mohamed</strong><br>
                <a href="mailto:youssefaymanmohamed1@gmail.com">Email</a> | <a href="https://github.com/youssefaymanmohamed" target="_blank">GitHub</a>
            </li>
            <li style="margin-bottom: 12px;">
                👨‍💻 <strong>Hazem Mohamed Abdelghafar</strong><br>
                <a href="mailto:hazem.metwalli23@gmail.com">Email</a> | <a href="https://github.com/HazemAbdelghafar" target="_blank">GitHub</a>
            </li>
            <li style="margin-bottom: 12px;">
                👩‍💻 <strong>Noureen Tamer Magdy</strong><br>
                <a href="mailto:noureentamer8@gmail.com">Email</a> | <a href="https://github.com/noureen-tamer" target="_blank">GitHub</a>
            </li>
            <li style="margin-bottom: 12px;">
                👨‍💻 <strong>Mariam Ahmed Mahmoud</strong><br>
                <a href="mailto:mariamahmedtalaat@gmail.com">Email</a> | <a href="https://github.com/MariamTalaat28" target="_blank">GitHub</a>
            </li>
            <li style="margin-bottom: 12px;">
                👩‍💻 <strong>Haneen Elsayed Ahmed</strong><br>
                <a href="mailto:haneenelbawab8@gmail.com">Email</a> | <a href="https://github.com/HaneenElbawab" target="_blank">GitHub</a>
            </li>
            <li style="margin-bottom: 12px;">
                👩‍💻 <strong>Hagar Mohamed Ibrahim</strong><br>
                <a href="mailto:hm294554@gmail.com">Email</a> | <a href="https://github.com/Hager205" target="_blank">GitHub</a>
            </li>
            </ul>
        </div>
        """
    
    st.sidebar.markdown(team_html, unsafe_allow_html=True)
    # st.sidebar.image(image2)
    st.title("Predicting Customer Churn")
    if add_selectbox == "Single entry":
        # Membership category
        membership_category = st.selectbox(
            "Membership Category:", 
            ["No Membership", "Silver Membership", "Gold Membership", 
             "Premium Membership", "Platinum Membership"]
        )
        
        # Gender
        gender = st.selectbox("Gender:", ["Male", "Female"])
        
        # Points in wallet
        points_in_wallet = st.number_input("Points in Wallet:", min_value=0.0, value=500.0, step=10.0)
        
        # Internet option
        internet_option = st.selectbox("Internet Option WiFi:", ["Yes", "No"])
        
        # Transaction value
        avg_transaction_value = st.number_input("Average Transaction Value:", min_value=0.0, value=10000.0, step=100.0)
        
        # Age
        age = st.number_input("Age:", min_value=10, max_value=100, value=30)
        
        # Used special discount
        offer_application_preference = st.selectbox("Offer application preference:", ["Yes", "No"])
        
        # Days since last login
        days_since_last_login = st.number_input("Days Since Last Login:", min_value=0, max_value=100, value=5)
        
        # Last visit time
        last_visit_time = st.time_input("Last Visit Time:", value=None)
        last_visit_hour = last_visit_time.hour if last_visit_time else 12
        
        # Feedback
        feedback = st.selectbox(
            "Feedback:", 
            ["Reasonable Price", "User Friendly Website", "Products Always in Stock", 
             "Poor Product Quality", "Quality Customer Care"]
        )
        
        # Joining date
        joining_date = st.date_input("Joining Date:")
        days_since_joining = (datetime.date.today() - joining_date).days if joining_date else 365
        joining_month = joining_date.month if joining_date else 6
        
        # Joined through referral
        joined_through_referral = st.selectbox("Joined Through Referral:", ["Yes", "No"])
        
        # Region category
        region_category = st.selectbox("Region Category Town:", ["Yes", "No"])
        
        # Average time spent
        avg_time_spent = st.number_input("Average Frequency Time Spent (minutes):", min_value=0.0, value=200.0, step=1.0)
        
        # Average login frequency
        avg_frequency_login_days = st.number_input("Average Frequency Login Days:", min_value=1, max_value=30, value=5)
        
        # Derived features (calculated automatically)
        joining_month_sin = math.sin(2 * math.pi * joining_month / 12)
        joining_month_cos = math.cos(2 * math.pi * joining_month / 12)
        last_visit_hour_sin = math.sin(2 * math.pi * last_visit_hour / 24)
        last_visit_hour_cos = math.cos(2 * math.pi * last_visit_hour / 24)
        interaction_frequency = 1 / avg_frequency_login_days
        recency_of_last_activity = days_since_last_login
        engagement_score = avg_time_spent * avg_transaction_value
        
        # Create input dictionary matching the features used in the model
        input_dict = {
            'points_in_wallet': points_in_wallet,
            'membership_category_Premium Membership': 1 if membership_category == "Premium Membership" else 0,
            'membership_category_Platinum Membership': 1 if membership_category == "Platinum Membership" else 0,
            'membership_category_No Membership': 1 if membership_category == "No Membership" else 0,
            'avg_transaction_value': avg_transaction_value,
            'membership_category_Gold Membership': 1 if membership_category == "Gold Membership" else 0,
            'engagment_score': engagement_score,
            'membership_category_Silver Membership': 1 if membership_category == "Silver Membership" else 0,
            'days_since_joining': days_since_joining,
            'avg_time_spent': avg_time_spent,
            'avg_frequency_login_days': avg_frequency_login_days,
            'interaction_frequency': interaction_frequency,
            'age': age,
            'feedback_Reasonable Price': 1 if feedback == "Reasonable Price" else 0,
            'feedback_User Friendly Website': 1 if feedback == "User Friendly Website" else 0,
            'feedback_Products always in Stock': 1 if feedback == "Products Always in Stock" else 0,
            'days_since_last_login': days_since_last_login,
            'recency_of_last_activity': recency_of_last_activity,
            'feedback_Quality Customer Care': 1 if feedback == "Quality Customer Care" else 0,
            'last_visit_hour_cos': last_visit_hour_cos,
            'last_visit_hour_sin': last_visit_hour_sin,
            'joining_month_sin': joining_month_sin,
            'joining_month_cos': joining_month_cos,
            'feedback_Poor Product Quality': 1 if feedback == "Poor Product Quality" else 0,
            'region_category_Town': 1 if region_category == "Yes" else 0,
            'joined_through_referral': 1 if joined_through_referral == "Yes" else 0,
            'gender': 1 if gender == "Male" else 0,
            'internet_option_Wi-Fi': 1 if internet_option == "Yes" else 0,
            'offer_application_preference': 1 if offer_application_preference == "Yes" else 0,
            'joining_year': joining_date.year if joining_date else 2023
        }

        trans = ['age', 'days_since_last_login', 'avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet','interaction_frequency','recency_of_last_activity', 'joining_year', 'days_since_joining']
        input = pd.DataFrame([input_dict.values()], columns=input_dict.keys())
        input[trans] = scaler.transform(input[trans])
        input = input.drop('joining_year', axis=1)

        if st.button("Predict"):
            # Custom animation with progress bar
            progress_text = "🔮 Analyzing customer data..."
            progress_bar = st.progress(0)
            
            # Simulate processing with incremental updates
            for percent_complete in range(0, 101, 20):
                time.sleep(0.1)  # Short delay for visual effect
                progress_bar.progress(percent_complete)
                if percent_complete == 0:
                    progress_text = "🔍 Scanning customer profile..."
                elif percent_complete == 20:
                    progress_text = "📊 Analyzing transaction patterns..."
                elif percent_complete == 40:
                    progress_text = "⏱️ Evaluating engagement metrics..."
                elif percent_complete == 60:
                    progress_text = "🔄 Processing behavioral data..."
                elif percent_complete == 80:
                    progress_text = "🧮 Calculating final prediction..."
                
                status_placeholder = st.empty()
                status_placeholder.text(progress_text)
            
            # Remove progress elements once complete
            progress_bar.empty()
            status_placeholder.empty()
            
            # Actual prediction logic
            y_pred = model.predict(input.values)

            st.success("Churn Prediction: {0}".format("Will Churn" if y_pred else "Will Not Churn"))

            # Log the prediction
            logger.info(f"Single prediction request: {input_dict} | Result: {'Will Churn' if y_pred else 'Will Not Churn'}")

    if add_selectbox == "Batch entry":
        st.subheader("Batch Prediction")
        st.write("Upload a CSV file with the same features as above.")
        uploaded_file = st.file_uploader("Choose a file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data Uploaded Successfully")
            st.write(data.head())
            data=data.drop('joining_year', axis=1)
            
            if st.button("Predict"):
                # Custom animation with progress bar
                progress_text = "🔮 Analyzing customer data..."
                progress_bar = st.progress(0)
                
                # Simulate processing with incremental updates
                for percent_complete in range(0, 101, 20):
                    time.sleep(0.1)  # Short delay for visual effect
                    progress_bar.progress(percent_complete)
                    if percent_complete == 0:
                        progress_text = "🔍 Scanning customer profile..."
                    elif percent_complete == 20:
                        progress_text = "📊 Analyzing transaction patterns..."
                    elif percent_complete == 40:
                        progress_text = "⏱️ Evaluating engagement metrics..."
                    elif percent_complete == 60:
                        progress_text = "🔄 Processing behavioral data..."
                    elif percent_complete == 80:
                        progress_text = "🧮 Calculating final prediction..."
                    
                    status_placeholder = st.empty()
                    status_placeholder.text(progress_text)
                
                # Remove progress elements once complete
                progress_bar.empty()
                status_placeholder.empty()
                
                # Actual prediction logic
                y_pred = model.predict(data.values)
                print(y_pred)

                for i, pred in enumerate(y_pred):
                    st.success("Churn Prediction for index {0}: {1}".format(i, "Will Churn" if pred else "Will Not Churn"))

                # Log the prediction
                logger.info(f"Batch prediction request: {data.to_dict(orient='records')} | Results: {y_pred.tolist()}")


if __name__ == "__main__":
    main()
