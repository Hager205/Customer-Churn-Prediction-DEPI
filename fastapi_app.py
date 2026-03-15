import math
import logging
import datetime
import pandas as pd
from pickle import load
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Mount static files (CSS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
# Set up templates directory
templates = Jinja2Templates(directory="app/templates")

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
    filename='fastapi_app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Define the expected input schema
class ChurnInput(BaseModel):
    points_in_wallet: float
    membership_category: str
    avg_transaction_value: float
    age: int
    feedback: str
    days_since_last_login: int
    avg_time_spent: float
    avg_frequency_login_days: int
    gender: str
    internet_option: str
    offer_application_preference: str
    joining_date: str  # ISO format date string
    last_visit_time: str  # HH:MM:SS format
    joined_through_referral: str
    region_category: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_churn(input: ChurnInput):
    try:
        # Parse dates and times
        joining_date = datetime.datetime.strptime(input.joining_date, "%Y-%m-%d").date()
        days_since_joining = (datetime.date.today() - joining_date).days
        joining_month = joining_date.month
        joining_year = joining_date.year
        last_visit_time = datetime.datetime.strptime(input.last_visit_time, "%H:%M:%S").time()
        last_visit_hour = last_visit_time.hour

        # Derived features
        joining_month_sin = math.sin(2 * math.pi * joining_month / 12)
        joining_month_cos = math.cos(2 * math.pi * joining_month / 12)
        last_visit_hour_sin = math.sin(2 * math.pi * last_visit_hour / 24)
        last_visit_hour_cos = math.cos(2 * math.pi * last_visit_hour / 24)
        interaction_frequency = 1 / input.avg_frequency_login_days
        recency_of_last_activity = input.days_since_last_login
        engagement_score = input.avg_time_spent * input.avg_transaction_value

        # One-hot and binary encodings
        input_dict = {
            'points_in_wallet': input.points_in_wallet,
            'membership_category_Premium Membership': 1 if input.membership_category == "Premium Membership" else 0,
            'membership_category_Platinum Membership': 1 if input.membership_category == "Platinum Membership" else 0,
            'membership_category_No Membership': 1 if input.membership_category == "No Membership" else 0,
            'avg_transaction_value': input.avg_transaction_value,
            'membership_category_Gold Membership': 1 if input.membership_category == "Gold Membership" else 0,
            'engagment_score': engagement_score,
            'membership_category_Silver Membership': 1 if input.membership_category == "Silver Membership" else 0,
            'days_since_joining': days_since_joining,
            'avg_time_spent': input.avg_time_spent,
            'avg_frequency_login_days': input.avg_frequency_login_days,
            'interaction_frequency': interaction_frequency,
            'age': input.age,
            'feedback_Reasonable Price': 1 if input.feedback == "Reasonable Price" else 0,
            'feedback_User Friendly Website': 1 if input.feedback == "User Friendly Website" else 0,
            'feedback_Products always in Stock': 1 if input.feedback == "Products Always in Stock" else 0,
            'days_since_last_login': input.days_since_last_login,
            'recency_of_last_activity': recency_of_last_activity,
            'feedback_Quality Customer Care': 1 if input.feedback == "Quality Customer Care" else 0,
            'last_visit_hour_cos': last_visit_hour_cos,
            'last_visit_hour_sin': last_visit_hour_sin,
            'joining_month_sin': joining_month_sin,
            'joining_month_cos': joining_month_cos,
            'feedback_Poor Product Quality': 1 if input.feedback == "Poor Product Quality" else 0,
            'region_category_Town': 1 if input.region_category == "Yes" else 0,
            'joined_through_referral': 1 if input.joined_through_referral == "Yes" else 0,
            'gender': 1 if input.gender == "Male" else 0,
            'internet_option_Wi-Fi': 1 if input.internet_option == "Yes" else 0,
            'offer_application_preference': 1 if input.offer_application_preference == "Yes" else 0,
            'joining_year': joining_year
        }

        trans = ['age', 'days_since_last_login', 'avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet','interaction_frequency','recency_of_last_activity', 'joining_year', 'days_since_joining']
        df = pd.DataFrame([input_dict.values()], columns=input_dict.keys())
        df[trans] = scaler.transform(df[trans])
        df = df.drop('joining_year', axis=1)

        y_pred = model.predict(df.values)
        result = "Will Churn" if y_pred[0] else "Will Not Churn"
        logger.info(f"Prediction request: {input.dict()} | Result: {result}")
        return {"prediction": result}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}