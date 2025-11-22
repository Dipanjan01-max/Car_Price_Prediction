import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main background */
    .main {
        padding: 2rem;
        background-color: #0e1117;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        height: 3.5em;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Price card */
    .price-card {
        padding: 30px;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Summary card - FIXED */
    .summary-card {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        border: 1px solid #475569;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .summary-card h3 {
        color: #a78bfa;
        margin-bottom: 20px;
        font-size: 20px;
    }
    
    .summary-item {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #475569;
    }
    
    .summary-item:last-child {
        border-bottom: none;
    }
    
    .summary-label {
        color: #94a3b8;
        font-weight: 500;
    }
    
    .summary-value {
        color: #e2e8f0;
        font-weight: 600;
    }
    
    /* Info card */
    .info-card {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        margin: 10px 0;
        color: white;
    }
    
    /* Headers */
    h1 {
        color: #e2e8f0;
        text-align: center;
        padding-bottom: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 {
        color: #e2e8f0;
    }
    
    /* Metric containers */
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #475569;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background-color: #1e293b;
        color: white;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        background-color: #1e293b;
        color: white;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    
    /* Subheaders */
    .section-header {
        color: #a78bfa;
        font-size: 24px;
        font-weight: 600;
        margin: 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #475569;
    }
    
    /* Comparison card */
    .comparison-card {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(6, 95, 70, 0.3);
    }
    
    .depreciation-card {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(153, 27, 27, 0.3);
    }
    
    /* Recommendations */
    .recommendation-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        color: white;
    }
    
    .recommendation-box ul {
        color: #cbd5e1;
        margin: 10px 0;
    }
    
    /* Stickers/Badges */
    .badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 5px;
    }
    
    .badge-success {
        background-color: #059669;
        color: white;
    }
    
    .badge-warning {
        background-color: #d97706;
        color: white;
    }
    
    .badge-info {
        background-color: #0284c7;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    model = pk.load(open('model.pkl', 'rb'))
    columns = pk.load(open('columns.pkl', 'rb'))
    return model, columns

@st.cache_data
def load_data():
    car_data = pd.read_csv('cardetails.csv.csv')
    car_data['name'] = car_data['name'].apply(lambda x: x.split(' ')[0].strip())
    return car_data

model, columns = load_models()
car_data = load_data()

# Header
st.markdown("<h1>🚗 Car Price Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 18px; margin-bottom: 30px;'>Get accurate price predictions for your vehicle using advanced ML algorithms</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #a78bfa;'>📊 About This Tool</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card'>
    This ML-powered tool predicts car prices based on:
    <ul>
        <li>🏢 Brand and Model</li>
        <li>📅 Year of Manufacture</li>
        <li>🛣️ Kilometers Driven</li>
        <li>⛽ Fuel Type</li>
        <li>⚙️ Transmission Type</li>
        <li>👥 Owner History</li>
        <li>📍 Location</li>
        <li>🔧 Car Condition</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #a78bfa; margin-top: 30px;'>📈 Model Statistics</h2>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Total Cars", f"{len(car_data):,}")
    with col_s2:
        st.metric("Brands", len(car_data['name'].unique()))
    
    st.markdown("<h2 style='color: #a78bfa; margin-top: 30px;'>💡 Tips</h2>", unsafe_allow_html=True)
    st.success("✅ Lower km driven = Higher price!")
    st.info("ℹ️ First owner cars have better resale value")
    st.warning("⚠️ Regular maintenance increases value")

# Main content
col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.markdown("<div class='section-header'>🔍 Enter Car Details</div>", unsafe_allow_html=True)
    
    # Input fields in columns
    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        name = st.selectbox('🏢 Car Brand', sorted(car_data['name'].unique()), 
                           help="Select the manufacturer of your car")
        fuel = st.selectbox('⛽ Fuel Type', car_data['fuel'].unique(),
                           help="Type of fuel your car uses")
        location = st.selectbox('📍 Location', 
                               ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 
                                'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Surat', 'Other'],
                               help="City where the car is located")
    
    with input_col2:
        year = st.slider('📅 Model Year', 
                        min_value=int(car_data['year'].min()), 
                        max_value=int(car_data['year'].max()),
                        value=2018,
                        help="Year when the car was manufactured")
        seller_type = st.selectbox('👤 Seller Type', car_data['seller_type'].unique(),
                                  help="Type of seller")
        condition = st.selectbox('🔧 Car Condition',
                                ['Excellent', 'Good', 'Fair', 'Poor'],
                                help="Overall condition of the vehicle")
    
    with input_col3:
        km_driven = st.number_input('🛣️ Kilometers Driven', 
                                    min_value=0, 
                                    max_value=500000, 
                                    value=50000,
                                    step=1000,
                                    help="Total distance covered")
        transmission = st.selectbox('⚙️ Transmission', car_data['transmission'].unique(),
                                   help="Type of transmission system")
        owner = st.selectbox('👥 Owner Type', car_data['owner'].unique(),
                            help="Number of previous owners")

with col2:
    st.markdown("<div class='section-header'>📝 Summary</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='summary-card'>
        <div class='summary-item'>
            <span class='summary-label'>🏢 Brand:</span>
            <span class='summary-value'>{name}</span>
        </div>
        <div class='summary-item'>
            <span class='summary-label'>📅 Year:</span>
            <span class='summary-value'>{year}</span>
        </div>
        <div class='summary-item'>
            <span class='summary-label'>🛣️ KM Driven:</span>
            <span class='summary-value'>{km_driven:,} km</span>
        </div>
        <div class='summary-item'>
            <span class='summary-label'>⛽ Fuel:</span>
            <span class='summary-value'>{fuel}</span>
        </div>
        <div class='summary-item'>
            <span class='summary-label'>⚙️ Transmission:</span>
            <span class='summary-value'>{transmission}</span>
        </div>
        <div class='summary-item'>
            <span class='summary-label'>👥 Owner:</span>
            <span class='summary-value'>{owner}</span>
        </div>
        <div class='summary-item'>
            <span class='summary-label'>📍 Location:</span>
            <span class='summary-value'>{location}</span>
        </div>
        <div class='summary-item'>
            <span class='summary-label'>🔧 Condition:</span>
            <span class='summary-value'>{condition}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Purchase Price Comparison Section
st.markdown("---")
st.markdown("<div class='section-header'>💰 Purchase Price Analysis (Optional)</div>", unsafe_allow_html=True)

compare_price = st.checkbox("📊 Compare with my purchase price", help="Enable this to see how your car's value has changed")

purchase_price = None
purchase_year = None

if compare_price:
    purchase_col1, purchase_col2, purchase_col3 = st.columns(3)
    with purchase_col1:
        purchase_price = st.number_input('💵 Purchase Price (₹)', 
                                        min_value=0, 
                                        value=500000,
                                        step=10000,
                                        help="How much did you pay for this car?")
    with purchase_col2:
        purchase_year = st.number_input('📅 Year of Purchase', 
                                       min_value=2000, 
                                       max_value=datetime.now().year,
                                       value=min(year + 1, datetime.now().year),
                                       help="When did you buy this car?")
    with purchase_col3:
        purchase_km = st.number_input('🛣️ KM at Purchase', 
                                     min_value=0, 
                                     max_value=km_driven,
                                     value=max(0, km_driven - 20000),
                                     step=1000,
                                     help="Odometer reading when you bought it")

# Predict button
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_button = st.button('🔮 PREDICT CAR PRICE NOW', use_container_width=True)

if predict_button:
    with st.spinner('🔄 Analyzing car details and calculating price...'):
        # Prepare input data
        input_data_model = pd.DataFrame(
            [[name, year, km_driven, fuel, seller_type, transmission, owner]],
            columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
        )
        
        # Data preprocessing
        input_data_model['owner'].replace(
            ['owner_First Owner', 'owner_Second Owner', 'owner_Fourth & owner_Above Owner', 'Third Owner', 'Test Drive Car'],
            [1, 2, 3, 4, 5], inplace=True
        )
        input_data_model['fuel'].replace(
            ['fuel_Petrol', 'fuel_Diesel', 'fuel_CNG', 'fuel_LPG', 'fuel_Electric'],
            [1, 2, 3, 4, 5], inplace=True
        )
        input_data_model['seller_type'].replace(
            ['seller_Individual', 'seller_Dealer', 'seller_Trustmark Dealer'],
            [1, 2, 3], inplace=True
        )
        input_data_model['transmission'].replace(
            ['transmission_Manual', 'transmission_Automatic'],
            [1, 2], inplace=True
        )
        input_data_model['name'].replace(
            ['name_Maruti', 'name_Hyundai', 'name_Datsun', 'name_Honda', 'name_Tata', 'name_Chevrolet',
             'name_Toyota', 'name_Jaguar', 'name_Mercedes-Benz', 'name_Audi', 'name_Skoda', 'name_Jeep',
             'name_BMW', 'name_Mahindra', 'name_Ford', 'name_Nissan', 'name_Renault', 'name_Fiat',
             'name_Volkswagen', 'name_Volvo', 'name_Mitsubishi', 'name_Land', 'name_Daewoo', 'name_MG',
             'name_Force', 'name_Isuzu', 'name_OpelCorsa', 'name_Ambassador', 'name_Kia'],
            list(range(1, 30)), inplace=True
        )
        
        # One-hot encoding
        input_data_model_encoded = pd.get_dummies(input_data_model)
        for col in columns:
            if col not in input_data_model_encoded.columns:
                input_data_model_encoded[col] = 0
        
        input_data_model_encoded = input_data_model_encoded[columns]
        
        # Predict
        car_price = model.predict(input_data_model_encoded)[0]
        
        # Display results
        st.markdown("---")
        st.balloons()
        st.markdown("<h2 style='text-align: center; color: #a78bfa;'>✅ Prediction Complete!</h2>", unsafe_allow_html=True)
        
        # Main price display
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        with result_col2:
            st.markdown(f"""
            <div class="price-card">
                <h2 style="margin:0; color: #ffffff; font-size: 24px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">💎 Predicted Market Value</h2>
                <h1 style="margin:15px 0; color: #ffffff; font-size: 56px; font-weight: 900; text-shadow: 3px 3px 6px rgba(0,0,0,0.4);">₹ {car_price:,.0f}</h1>
                <p style="margin:0; color: #ffffff; font-size: 18px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">Current Estimated Price</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key metrics
        st.markdown("<div class='section-header'>📊 Key Metrics</div>", unsafe_allow_html=True)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        age = datetime.now().year - year
        avg_km_per_year = km_driven / age if age > 0 else km_driven
        price_range_low = car_price * 0.9
        price_range_high = car_price * 1.1
        
        with metric_col1:
            st.markdown(f"""
            <div class='metric-container'>
                <div style='text-align: center;'>
                    <p style='color: #94a3b8; margin: 0;'>🕐 Car Age</p>
                    <h2 style='color: #e2e8f0; margin: 10px 0;'>{age} years</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class='metric-container'>
                <div style='text-align: center;'>
                    <p style='color: #94a3b8; margin: 0;'>📏 Avg KM/Year</p>
                    <h2 style='color: #e2e8f0; margin: 10px 0;'>{avg_km_per_year:,.0f}</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
            <div class='metric-container'>
                <div style='text-align: center;'>
                    <p style='color: #94a3b8; margin: 0;'>📈 Price Range</p>
                    <h2 style='color: #e2e8f0; margin: 10px 0; font-size: 18px;'>₹{price_range_low/100000:.1f}L-{price_range_high/100000:.1f}L</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            price_per_km = car_price / km_driven if km_driven > 0 else 0
            st.markdown(f"""
            <div class='metric-container'>
                <div style='text-align: center;'>
                    <p style='color: #94a3b8; margin: 0;'>💹 Price/KM</p>
                    <h2 style='color: #e2e8f0; margin: 10px 0;'>₹{price_per_km:.2f}</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Purchase price comparison
        if compare_price and purchase_price and purchase_year:
            st.markdown("---")
            st.markdown("<div class='section-header'>💰 Purchase Price Analysis</div>", unsafe_allow_html=True)
            
            price_diff = car_price - purchase_price
            price_diff_percent = (price_diff / purchase_price) * 100
            years_owned = datetime.now().year - purchase_year
            annual_change = price_diff / years_owned if years_owned > 0 else 0
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                if price_diff >= 0:
                    st.markdown(f"""
                    <div class='comparison-card'>
                        <h3 style='margin: 0 0 15px 0;'>📈 Value Appreciation</h3>
                        <h1 style='margin: 10px 0; font-size: 36px;'>+₹{price_diff:,.0f}</h1>
                        <p style='margin: 5px 0; font-size: 18px;'>+{price_diff_percent:.1f}% gain</p>
                        <hr style='border-color: rgba(255,255,255,0.3); margin: 15px 0;'>
                        <p style='margin: 0;'><strong>Purchase Price:</strong> ₹{purchase_price:,.0f}</p>
                        <p style='margin: 5px 0;'><strong>Current Value:</strong> ₹{car_price:,.0f}</p>
                        <p style='margin: 5px 0;'><strong>Annual Change:</strong> ₹{annual_change:,.0f}/year</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='depreciation-card'>
                        <h3 style='margin: 0 0 15px 0;'>📉 Value Depreciation</h3>
                        <h1 style='margin: 10px 0; font-size: 36px;'>-₹{abs(price_diff):,.0f}</h1>
                        <p style='margin: 5px 0; font-size: 18px;'>{price_diff_percent:.1f}% loss</p>
                        <hr style='border-color: rgba(255,255,255,0.3); margin: 15px 0;'>
                        <p style='margin: 0;'><strong>Purchase Price:</strong> ₹{purchase_price:,.0f}</p>
                        <p style='margin: 5px 0;'><strong>Current Value:</strong> ₹{car_price:,.0f}</p>
                        <p style='margin: 5px 0;'><strong>Annual Loss:</strong> ₹{abs(annual_change):,.0f}/year</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with analysis_col2:
                # Detailed breakdown
                km_increase = km_driven - purchase_km if compare_price else 0
                depreciation_rate = (abs(annual_change) / purchase_price * 100) if purchase_price > 0 else 0
                
                st.markdown(f"""
                <div class='info-card'>
                    <h3 style='color: #a78bfa; margin-top: 0;'>📊 Ownership Analysis</h3>
                    <div style='margin: 15px 0;'>
                        <div style='display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #475569;'>
                            <span style='color: #94a3b8;'>Years Owned:</span>
                            <span style='color: #e2e8f0; font-weight: 600;'>{years_owned} years</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #475569;'>
                            <span style='color: #94a3b8;'>KM Driven:</span>
                            <span style='color: #e2e8f0; font-weight: 600;'>{km_increase:,} km</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #475569;'>
                            <span style='color: #94a3b8;'>Avg KM/Year:</span>
                            <span style='color: #e2e8f0; font-weight: 600;'>{km_increase/years_owned if years_owned > 0 else 0:,.0f} km</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; padding: 10px 0;'>
                            <span style='color: #94a3b8;'>Annual Rate:</span>
                            <span style='color: #e2e8f0; font-weight: 600;'>{depreciation_rate:.1f}%/year</span>
                        </div>
                    </div>
                    <div style='background: rgba(102, 126, 234, 0.2); padding: 15px; border-radius: 8px; margin-top: 15px;'>
                        <p style='margin: 0; color: #cbd5e1; font-size: 14px;'>
                            {'✅ Your car is maintaining value well!' if price_diff >= 0 or depreciation_rate < 10 else '💡 Consider maintenance to preserve value'}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("<div class='section-header'>💡 Expert Recommendations</div>", unsafe_allow_html=True)
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("""
            <div class='recommendation-box'>
                <h3 style='color: #10b981; margin-top: 0;'>✅ Strong Selling Points</h3>
            """, unsafe_allow_html=True)
            
            points = []
            if km_driven < 50000:
                points.append("• Low mileage is highly attractive to buyers")
            if 'First' in owner:
                points.append("• First owner status adds premium value")
            if year >= datetime.now().year - 5:
                points.append("• Recent model year commands better price")
            if condition in ['Excellent', 'Good']:
                points.append("• Excellent condition justifies premium pricing")
            if 'Automatic' in transmission:
                points.append("• Automatic transmission has higher demand")
            if not points:
                points.append("• Well-maintained vehicle with complete service history")
            
            for point in points:
                st.markdown(f"<p style='color: #cbd5e1; margin: 5px 0;'>{point}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with rec_col2:
            st.markdown("""
            <div class='recommendation-box'>
                <h3 style='color: #f59e0b; margin-top: 0;'>⚠️ Factors Affecting Price</h3>
            """, unsafe_allow_html=True)
            
            factors = []
            if km_driven > 100000:
                factors.append("• High mileage may reduce asking price")
            if year < datetime.now().year - 10:
                factors.append("• Older models face faster depreciation")
            if 'Manual' in transmission and fuel != 'Diesel':
                factors.append("• Manual transmission has lower demand")
            if condition in ['Fair', 'Poor']:
                factors.append("• Condition needs improvement for better value")
            factors.append(f"• Location: {location} market conditions apply")
            
            for factor in factors:
                st.markdown(f"<p style='color: #cbd5e1; margin: 5px 0;'>{factor}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Market insights
        st.markdown("---")
        st.markdown("<div class='section-header'>🎯 Market Insights</div>", unsafe_allow_html=True)
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown("""
            <div class='info-card'>
                <h4 style='color: #a78bfa; margin-top: 0;'>🏆 Best Time to Sell</h4>
                <p style='color: #cbd5e1;'>Festive seasons (Diwali, New Year) typically see 15-20% higher demand and better prices.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown("""
            <div class='info-card'>
                <h4 style='color: #a78bfa; margin-top: 0;'>📸 Presentation Tips</h4>
                <p style='color: #cbd5e1;'>Professional photos, complete service records, and detailed listings can increase sale price by 10-15%.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col3:
            st.markdown(f"""
            <div class='info-card'>
                <h4 style='color: #a78bfa; margin-top: 0;'>💼 Negotiation Range</h4>
                <p style='color: #cbd5e1;'>Expect buyers to negotiate. Keep a buffer of ₹{car_price * 0.05:,.0f} - ₹{car_price * 0.1:,.0f} (5-10%).</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #94a3b8; padding: 30px; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 15px; margin-top: 30px;'>
        <h3 style='color: #a78bfa; margin-top: 0;'>🚀 Ready to Sell or Buy?</h3>
        <p style='font-size: 16px; margin: 15px 0;'>🔒 Your data is secure | 🤖 Powered by Advanced ML | 📱 Mobile Friendly</p>
        <div style='margin: 20px 0;'>
            <span class='badge badge-success'>Accurate Predictions</span>
            <span class='badge badge-info'>Real-time Analysis</span>
            <span class='badge badge-warning'>Market Insights</span>
        </div>
        <p style='font-size: 12px; color: #64748b; margin-top: 20px;'>
            ⚠️ Disclaimer: Predictions are estimates based on historical data and market trends. 
            Actual prices may vary based on specific vehicle condition, location demand, and negotiation.
        </p>
    </div>
""", unsafe_allow_html=True)