🚗 Car Price Prediction ML App

A Machine Learning web application built with Streamlit that predicts the price of a car based on its features such as brand, year, fuel type, seller type, transmission, and ownership.

This project uses a trained ML model (model.pkl) and encoded feature columns (columns.pkl) to make accurate price predictions.

📌 Features

Simple, user-friendly Streamlit interface

Select car brand, year, fuel type, seller type, transmission, ownership, and kilometers driven

Automatic encoding of inputs to match the trained model

Real-time price prediction using the loaded ML model

Supports multiple car brands and fuel types

🧰 Tech Stack
Component	Technology
Frontend	Streamlit
Backend	Python (pandas, numpy)
ML Model	scikit-learn
Data	cardetails.csv
📦 Installation
1️⃣ Clone the repository
git clone <your-repo-url>
cd Car_Price_prediction

🏗 Create a Virtual Environment (Python 3.11)
py -3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip

📚 Install Dependencies
pip install pandas numpy scikit-learn streamlit pyarrow


(You don't need to install pickle — it is built into Python.)

▶️ Run the Streamlit App
streamlit run car_price_prediction.py


Replace the filename if your file has a different name.

📁 Project Structure
Car_Price_prediction/
│
├── car_price_prediction.py   # Streamlit UI + prediction logic
├── model.pkl                 # Trained ML model
├── columns.pkl               # Columns used in model training
├── cardetails.csv            # Raw car dataset
├── venv/                     # Virtual environment
└── README.md                 # Project documentation

🧠 How It Works

The app loads:

The trained ML model (model.pkl)

The list of encoded columns (columns.pkl)

The dataset to extract unique dropdown values

User selects car features through the UI.

The app:

Encodes the categorical values

Reorders columns to match the trained model

Predicts the price using the ML model

The predicted price is displayed instantly.
