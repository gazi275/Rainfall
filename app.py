import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import json
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.cluster import KMeans

# ======= Page Configuration =======
st.set_page_config(
    page_title="Rainfall Analysis & Forecasting - Bangladesh",
    page_icon="☔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======= Custom Styling =======
st.markdown("""
    <style>
        .navbar {
            background-color: #003049;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        .navbar-title {
            font-size: 24px;
            font-weight: bold;
        }
        .navbar-links a {
            color: white;
            margin-left: 30px;
            text-decoration: none;
            font-size: 16px;
        }
        .navbar-links a:hover {
            text-decoration: underline;
        }
        .banner {
            background: linear-gradient(to right, #669bbc, #a1cbe6);
            padding: 2rem;
            text-align: center;
            border-radius: 0.5rem;
            color: white;
            margin: 20px 0;
        }
        .banner h1 {
            margin-bottom: 0.5rem;
            font-size: 38px;
            font-weight: bold;
        }
        .banner p {
            font-size: 18px;
        }
        .section {
            padding: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="navbar">
    <div class="navbar-title">Rainfall AI Dashboard</div>
    <div class="navbar-links">
        <a href="#overview">Overview</a>
        <a href="#visualizations">Visualizations</a>
        <a href="#models">Models</a>
        <a href="#forecast">Forecast</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="banner">
    <h1>☔️ Rainfall Analysis & Forecasting (Bangladesh)</h1>
    <p>Explore historical rainfall trends, seasonal patterns, and 10-year forecasts powered by machine learning and time series models.</p>
</div>
""", unsafe_allow_html=True)

# ======= Data Loading and Preprocessing =======
@st.cache_data
def load_data():
    main_path = './data/bgd-rainfall-adm2-full.csv'
    shapefile_path = './data/adm2Shape/bgd_admbnda_adm2_bbs_20201113.shp'
    try:
        data = pd.read_csv(main_path, low_memory=False)
        gdf = gpd.read_file(shapefile_path)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None
    
    # Debug: Check if ADM2_PCODE exists
    if 'ADM2_PCODE' not in data.columns:
        st.error("Column 'ADM2_PCODE' not found in the rainfall data. Please check the CSV file.")
        return None, None, None
    
    if 'ADM2_PCODE' not in gdf.columns:
        st.error("Column 'ADM2_PCODE' not found in the shapefile. Please check the shapefile.")
        return None, None, None
    
    # Merge with shapefile
    data = data.merge(gdf[['ADM2_PCODE', 'ADM2_EN']], on='ADM2_PCODE', how='left')
    
    # Debug: Check if merge was successful
    if data['ADM2_EN'].isna().all():
        st.warning("Merge with shapefile failed: No matching ADM2_PCODE values found.")
    
    # Date parsing and cleaning
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='coerce')
    data['rfh'] = pd.to_numeric(data['rfh'], errors='coerce')
    data = data.dropna(subset=['date', 'rfh'])
    
    # Extract time features
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['quarter'] = data['date'].dt.quarter
    data['is_monsoon'] = data['month'].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)
    
    # Define season mapping
    season_mapping = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon"
    }
    data['season'] = data['month'].map(season_mapping)
    
    # Add lag and rolling features
    data['rfh_lag1'] = data.groupby('ADM2_PCODE')['rfh'].shift(1)
    data['rfh_lag2'] = data.groupby('ADM2_PCODE')['rfh'].shift(2)
    data['rfh_roll3'] = data.groupby('ADM2_PCODE')['rfh'].rolling(3).mean().reset_index(0, drop=True)
    data['rfh_roll6'] = data.groupby('ADM2_PCODE')['rfh'].rolling(6).mean().reset_index(0, drop=True)
    data['rfh_diff'] = data.groupby('ADM2_PCODE')['rfh'].diff()
    
    # Clean lags
    data[['rfh_lag1', 'rfh_lag2', 'rfh_roll3', 'rfh_roll6', 'rfh_diff']] = data[['rfh_lag1', 'rfh_lag2', 'rfh_roll3', 'rfh_roll6', 'rfh_diff']].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    
    # Store original data with ADM2_PCODE for visualizations
    data_original = data.copy()
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['season', 'ADM2_PCODE'], drop_first=True)
    
    return data, gdf, data_original

data, gdf, data_original = load_data()
if data is None or gdf is None:
    st.stop()

def generate_dynamic_forecast(model, features, future_df, lag_columns, n_periods):
    forecast = np.zeros(n_periods)
    last_data = data_original[features + lag_columns].iloc[-1].copy()  # Initial last data as Series
    for i in range(n_periods):
        X_pred = future_df[features].iloc[i:i+1].copy()
        # Extract values for lag_columns only in the correct order
        lag_values = last_data[lag_columns].to_numpy()  # Ensure only lag_columns values
        if len(lag_values) != len(lag_columns):
            raise ValueError(f"Expected {len(lag_columns)} lag values, got {len(lag_values)}")
        lag_data_df = pd.DataFrame([lag_values[:len(lag_columns)]], index=X_pred.index, columns=lag_columns)
        X_pred[lag_columns] = lag_data_df
        pred = model.predict(X_pred)
        forecast[i] = pred[0]
        # Update last_data with new prediction
        last_data['rfh'] = pred[0]  # Temporary storage for rolling
        last_data['rfh_lag1'] = pred[0]
        last_data['rfh_lag2'] = last_data.get('rfh_lag1', pred[0])
        last_data['rfh_roll3'] = np.mean([pred[0], last_data.get('rfh_lag1', 0), last_data.get('rfh_lag2', 0)])
        last_data['rfh_roll6'] = last_data['rfh_roll3']  # Approx for 6-month rolling
        last_data['rfh_diff'] = pred[0] - last_data.get('rfh_lag1', 0)
    return forecast

# ======= Overview Section =======
st.markdown('<div class="section" id="overview">', unsafe_allow_html=True)
st.markdown("""
### 📃 Overview
This dashboard provides a comprehensive analysis of rainfall patterns in Bangladesh using district-wise data. It leverages advanced machine learning and time series models to predict future rainfall trends, supporting climate-smart agriculture and disaster preparedness.

**Included Models:** XGBoost, Random Forest, LightGBM, ARIMA, SARIMA, Prophet, LSTM
""")
st.markdown('</div>', unsafe_allow_html=True)

# ======= Visualizations Section =======
st.markdown('<div class="section" id="visualizations">', unsafe_allow_html=True)
st.markdown("""
### 🔍 Visualizations
Explore district-wise and seasonal rainfall patterns interactively using the options below.
""")

viz_option = st.selectbox("Choose a visualization", [
    "District-wise Rainfall Map",
    "Seasonal Variation",
    "Yearly Rainfall Trend",
    "Rainfall Clustering"
])

if viz_option == "District-wise Rainfall Map":
    year = st.slider("Select Year", min_value=int(data_original['year'].min()), max_value=int(data_original['year'].max()), value=2020)
    season_option = st.selectbox("Select Season", ["All", "Winter", "Summer", "Monsoon", "Post-Monsoon"])
    
    filtered_data = data_original[data_original['year'] == year]
    if season_option != "All":
        filtered_data = filtered_data[filtered_data['season'] == season_option]
    
    # Debug: Check if ADM2_PCODE exists
    if 'ADM2_PCODE' not in filtered_data.columns:
        st.error("Column 'ADM2_PCODE' missing in filtered data. Please check the data preprocessing steps.")
        st.stop()
    
    rainfall_summary = filtered_data.groupby('ADM2_PCODE')['rfh'].sum().reset_index()
    merged_gdf = gdf.merge(rainfall_summary, on='ADM2_PCODE', how='left')
    merged_gdf['rfh'] = merged_gdf['rfh'].fillna(0)
    merged_gdf['hover'] = merged_gdf['ADM2_EN'] + '<br>Rainfall: ' + merged_gdf['rfh'].astype(str) + ' mm'
    
    for col in merged_gdf.select_dtypes(include=['datetime64']).columns:
        merged_gdf[col] = merged_gdf[col].astype(str)
    
    fig_map = go.Figure(go.Choropleth(
        geojson=json.loads(merged_gdf.to_json()),
        featureidkey="properties.ADM2_PCODE",
        locations=merged_gdf['ADM2_PCODE'],
        z=merged_gdf['rfh'],
        colorscale="Blues",
        marker_opacity=0.7,
        marker_line_width=0,
        customdata=merged_gdf['hover'],
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5.5,
        mapbox_center={"lat": 23.685, "lon": 90.3563},
        margin={"r":0,"t":40,"l":0,"b":0},
        title=f"Rainfall in {season_option} {year} by District"
    )
    st.plotly_chart(fig_map, use_container_width=True)

elif viz_option == "Seasonal Variation":
    seasonal_data = data_original.groupby(['year', 'season'])['rfh'].mean().reset_index()
    fig = px.line(seasonal_data, x='year', y='rfh', color='season',
                  title='Average Rainfall by Season',
                  labels={'rfh': 'Rainfall (mm)', 'year': 'Year'},
                  color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

elif viz_option == "Yearly Rainfall Trend":
    yearly_data = data_original.groupby('year')['rfh'].sum().reset_index()
    fig = px.line(yearly_data, x='year', y='rfh',
                  title='Total Yearly Rainfall',
                  labels={'rfh': 'Rainfall (mm)', 'year': 'Year'},
                  color_discrete_sequence=['#003049'])
    st.plotly_chart(fig, use_container_width=True)

elif viz_option == "Rainfall Clustering":
    district_rainfall = data_original.groupby('ADM2_PCODE')['rfh'].mean().reset_index()
    kmeans = KMeans(n_clusters=4, random_state=42)
    district_rainfall['cluster'] = kmeans.fit_predict(district_rainfall[['rfh']])
    merged_gdf = gdf.merge(district_rainfall, on='ADM2_PCODE', how='left')
    merged_gdf['cluster'] = merged_gdf['cluster'].fillna(-1).astype(int)
    for col in merged_gdf.select_dtypes(include=['datetime64']).columns:
        merged_gdf[col] = merged_gdf[col].astype(str)
    
    fig = go.Figure(go.Choropleth(
        geojson=json.loads(merged_gdf.to_json()),
        featureidkey="properties.ADM2_PCODE",
        locations=merged_gdf['ADM2_PCODE'],
        z=merged_gdf['cluster'],
        colorscale="Viridis",
        marker_opacity=0.7,
        marker_line_width=0,
        customdata=merged_gdf['ADM2_EN'] + '<br>Cluster: ' + merged_gdf['cluster'].astype(str),
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5.5,
        mapbox_center={"lat": 23.685, "lon": 90.3563},
        margin={"r":0,"t":40,"l":0,"b":0},
        title="Rainfall Clustering by District"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ======= Models & Performance Section =======
st.markdown('<div class="section" id="models">', unsafe_allow_html=True)
st.markdown("""
### 🧪 Models & Performance
Compare the performance of different models for rainfall prediction using MAE, RMSE, and R² metrics, along with actual vs. predicted plots.
""")

@st.cache_resource
def train_models():
    results = {}
    
    # Define features
    features = ['year', 'month', 'quarter', 'is_monsoon', 'rfh_lag1', 'rfh_lag2', 'rfh_roll3', 'rfh_roll6', 'rfh_diff'] + \
               [col for col in data.columns if col.startswith('season_') or col.startswith('ADM2_PCODE_')]
    X = data[features].astype(float)
    y = data['rfh']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=7, min_child_weight=3,
                                 subsample=0.9, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X_train, y_train)
    results["XGBoost"] = (y_test, xgb_model.predict(X_test))
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    results["Random Forest"] = (y_test, rf_model.predict(X_test))
    
    # LightGBM
    lgbm_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
    lgbm_model.fit(X_train, y_train)
    results["LightGBM"] = (y_test, lgbm_model.predict(X_test))
    
    # ARIMA (for Dhaka, yearly data)
    target_district = "Dhaka"
    district_data = data_original[data_original['ADM2_EN'] == target_district].groupby('year')['rfh'].mean()
    district_data.index = pd.to_datetime(district_data.index, format='%Y')  # Set index as datetime
    train_size = int(len(district_data) * 0.8)
    train, test = district_data.iloc[:train_size], district_data.iloc[train_size:]
    arima_model = ARIMA(train, order=(2, 1, 2))
    arima_fit = arima_model.fit()
    results["ARIMA"] = (test, arima_fit.forecast(steps=len(test)))
    
    # SARIMA (for Dhaka, monthly data)
    district_data_monthly = data_original[data_original['ADM2_EN'] == target_district].set_index('date')['rfh'].resample('ME').mean()
    train_size = int(len(district_data_monthly) * 0.8)
    train, test = district_data_monthly.iloc[:train_size], district_data_monthly.iloc[train_size:]
    sarima_model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    results["SARIMA"] = (test, sarima_fit.forecast(steps=len(test)))
    
    # Prophet
    prophet_data = data_original.groupby('date')['rfh'].mean().reset_index()
    prophet_data.columns = ['ds', 'y']
    train_size = int(len(prophet_data) * 0.8)
    train, test = prophet_data.iloc[:train_size], prophet_data.iloc[train_size:]
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(train)
    future = prophet_model.make_future_dataframe(periods=len(test), freq='ME')
    forecast = prophet_model.predict(future)
    results["Prophet"] = (test['y'], forecast['yhat'].iloc[-len(test):].values)
    
    # LSTM
    sequence_features = ['rfh', 'rfh_lag1', 'rfh_lag2']
    sequence_data = data_original[sequence_features].copy().dropna()
    split_idx = int(len(sequence_data) * 0.8)
    train_seq = sequence_data.iloc[:split_idx]
    test_seq = sequence_data.iloc[split_idx:]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_seq)
    test_scaled = scaler.transform(test_seq)
    
    def create_sequences(scaled_data, seq_len=3):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(scaled_data)):
            X_seq.append(scaled_data[i-seq_len:i, :])
            y_seq.append(scaled_data[i, 0])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_lstm, y_train_lstm = create_sequences(train_scaled)
    X_test_lstm, y_test_lstm = create_sequences(test_scaled)
    
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(3, 3)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
    
    y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
    y_pred_lstm = scaler.inverse_transform(
        np.concatenate([y_pred_lstm_scaled, np.zeros((len(y_pred_lstm_scaled), 2))], axis=1)
    )[:, 0]
    y_test_lstm = scaler.inverse_transform(
        np.concatenate([y_test_lstm.reshape(-1, 1), np.zeros((len(y_test_lstm), 2))], axis=1)
    )[:, 0]
    results["LSTM"] = (y_test_lstm, y_pred_lstm)
    
    return results, X_train, X_test, y_train, y_test

models, X_train, X_test, y_train, y_test = train_models()

model_choice = st.selectbox("Select Model", list(models.keys()))

# Evaluation
y_true, y_pred = models[model_choice]
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
acc = 100 - (mae / np.mean(y_true) * 100) if np.mean(y_true) != 0 else 0

st.write(f"**{model_choice} Performance:**")
st.write(f"MAE: {mae:.2f} mm")
st.write(f"RMSE: {rmse:.2f} mm")
st.write(f"R²: {r2:.2f}")
st.write(f"Accuracy: {acc:.2f}%")

fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual Rainfall (mm)', 'y': 'Predicted Rainfall (mm)'},
                 title=f"{model_choice}: Actual vs Predicted")
fig.add_scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], mode='lines', name='Ideal', line=dict(color='red', dash='dash'))
st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ======= Forecasting Section =======
st.markdown('<div class="section" id="forecast">', unsafe_allow_html=True)
st.markdown("""
### ⏲️ Forecasting (2025-2035)
Predict future rainfall trends for Bangladesh using selected models. Compare forecasts across models for the next 10 years.
""")

# District selection for ARIMA/SARIMA
districts = sorted(data_original['ADM2_EN'].dropna().unique())
selected_district = st.selectbox("Select District for ARIMA/SARIMA", districts, index=districts.index("Dhaka") if "Dhaka" in districts else 0)
selected_models = st.multiselect("Select Forecast Models", ["XGBoost", "Random Forest", "LightGBM", "ARIMA", "SARIMA", "Prophet", "LSTM"], default=["Prophet", "ARIMA", "SARIMA"])

forecast_df_all = pd.DataFrame({"Date": pd.date_range(start='2025-01-01', end='2035-12-01', freq='ME')})

@st.cache_resource
def train_forecast_models(selected_district):
    forecast_results = {}
    
    # Define features
    features = ['year', 'month', 'quarter', 'is_monsoon', 'rfh_lag1', 'rfh_lag2', 'rfh_roll3', 'rfh_roll6', 'rfh_diff'] + \
               [col for col in data.columns if col.startswith('season_') or col.startswith('ADM2_PCODE_')]
    X = data[features].astype(float)
    y = data['rfh']
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=7, min_child_weight=3,
                                 subsample=0.9, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X, y)
    forecast_results["XGBoost"] = xgb_model
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X, y)
    forecast_results["Random Forest"] = rf_model
    
    # LightGBM
    lgbm_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
    lgbm_model.fit(X, y)
    forecast_results["LightGBM"] = lgbm_model
    
    # ARIMA (yearly data for selected district)
    district_data = data_original[data_original['ADM2_EN'] == selected_district].groupby('year')['rfh'].mean()
    district_data.index = pd.to_datetime(district_data.index, format='%Y')  # Set index as datetime
    train_size = int(len(district_data) * 0.8)
    train, test = district_data.iloc[:train_size], district_data.iloc[train_size:]
    arima_model = ARIMA(train, order=(2, 1, 2))
    arima_fit = arima_model.fit()
    forecast_results["ARIMA"] = arima_fit
    
    # SARIMA (monthly data for selected district)
    district_data_monthly = data_original[data_original['ADM2_EN'] == selected_district].set_index('date')['rfh'].resample('ME').mean()
    train_size = int(len(district_data_monthly) * 0.8)
    train, test = district_data_monthly.iloc[:train_size], district_data_monthly.iloc[train_size:]
    sarima_model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    forecast_results["SARIMA"] = sarima_fit
    
    # Prophet
    prophet_data = data_original.groupby('date')['rfh'].mean().reset_index()
    prophet_data.columns = ['ds', 'y']
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(prophet_data)
    forecast_results["Prophet"] = prophet_model
    
    # LSTM
    sequence_features = ['rfh', 'rfh_lag1', 'rfh_lag2']
    sequence_data = data_original[sequence_features].copy().dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(sequence_data)
    
    def create_sequences(scaled_data, seq_len=3):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(scaled_data)):
            X_seq.append(scaled_data[i-seq_len:i, :])
            y_seq.append(scaled_data[i, 0])
        return np.array(X_seq), np.array(y_seq)
    
    X_lstm, y_lstm = create_sequences(scaled_data)
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(3, 3)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
    forecast_results["LSTM"] = (lstm_model, scaler)
    
    return forecast_results

forecast_models = train_forecast_models(selected_district)

# Forecasting loop
for model_name in selected_models:
    future_dates = pd.date_range(start='2025-01-01', end='2035-12-01', freq='ME')
    future_df = pd.DataFrame(index=future_dates)
    future_df['year'] = future_df.index.year
    future_df['month'] = future_df.index.month
    future_df['quarter'] = future_df.index.quarter
    future_df['is_monsoon'] = future_df['month'].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)
    
    # Define lag_columns
    lag_columns = ['rfh_lag1', 'rfh_lag2', 'rfh_roll3', 'rfh_roll6', 'rfh_diff']
    
    # Initialize with last known lag values
    last_row = data_original[lag_columns].iloc[-1]  # Use only lag_columns from data_original
    future_df[lag_columns] = pd.DataFrame([last_row] * len(future_df), index=future_df.index, columns=lag_columns)
    
    # Add season and district dummies
    season_mapping = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon"
    }
    season_dummies = pd.get_dummies(future_df['month'].map(season_mapping), prefix='season').reindex(
        columns=[col for col in data_original.columns if col.startswith('season_')], fill_value=0)
    district_dummies = pd.DataFrame(0, index=future_df.index, columns=[col for col in data_original.columns if col.startswith('ADM2_PCODE_')])
    district_col = f"ADM2_PCODE_{data_original[data_original['ADM2_EN'] == selected_district]['ADM2_PCODE'].iloc[0]}"
    if district_col in district_dummies.columns:
        district_dummies[district_col] = 1
    future_df = pd.concat([future_df, season_dummies, district_dummies], axis=1)
    
    if model_name in ["XGBoost", "Random Forest", "LightGBM"]:
        features = ['year', 'month', 'quarter', 'is_monsoon', 'rfh_lag1', 'rfh_lag2', 'rfh_roll3', 'rfh_roll6', 'rfh_diff'] + \
                   [col for col in future_df.columns if col.startswith('season_') or col.startswith('ADM2_PCODE_')]
        last_data = data_original[features].iloc[-1].copy()  # Use full features for last_data
        forecast = generate_dynamic_forecast(forecast_models[model_name], features, future_df, lag_columns, len(future_dates))
    
    elif model_name == "ARIMA":
        forecast_index = pd.date_range(start='2025-01-01', periods=len(future_dates), freq='YE')
        forecast = forecast_models[model_name].forecast(steps=len(future_dates))
        forecast = pd.Series(forecast, index=forecast_index)
        forecast = forecast.reindex(future_dates, method='ffill').values
    
    elif model_name == "SARIMA":
        forecast_index = pd.date_range(start='2025-01-01', periods=len(future_dates), freq='ME')
        forecast = forecast_models[model_name].forecast(steps=len(future_dates))
        forecast = pd.Series(forecast, index=forecast_index)
        forecast = forecast.reindex(future_dates, method='ffill').values
    
    elif model_name == "Prophet":
        future = forecast_models[model_name].make_future_dataframe(periods=len(future_dates), freq='ME')
        forecast_result = forecast_models[model_name].predict(future)
        forecast = forecast_result['yhat'].iloc[-len(future_dates):].values
    
    elif model_name == "LSTM":
        lstm_model, scaler = forecast_models[model_name]
        sequence_data = data_original[['rfh', 'rfh_lag1', 'rfh_lag2']].copy().dropna()
        scaled_data = scaler.transform(sequence_data.iloc[-3:])
        X_lstm = [scaled_data]
        forecast = []
        for _ in range(len(future_dates)):
            X_lstm_array = np.array(X_lstm[-1]).reshape(1, 3, 3)
            pred_scaled = lstm_model.predict(X_lstm_array, verbose=0)
            pred = scaler.inverse_transform(np.concatenate([pred_scaled, np.zeros((1, 2))], axis=1))[:, 0][0]
            forecast.append(pred)
            new_row = np.array([[pred, X_lstm[-1][-1, 0], X_lstm[-1][-1, 1]]])
            new_row_scaled = scaler.transform(new_row)
            X_lstm.append(np.vstack([X_lstm[-1][1:], new_row_scaled]))
            if len(forecast) > 10 and np.std(forecast[-10:]) < 0.1:
                break
        forecast = np.array(forecast[:len(future_dates)])
    
    forecast_df_all[model_name] = forecast

# Plot Forecasts
st.subheader("📈 Forecast Comparison")
fig = px.line(forecast_df_all, x="Date", y=selected_models, title=f"Forecasted Rainfall for {selected_district} (2025–2035)",
              labels={"value": "Rainfall (mm)", "variable": "Model"})
st.plotly_chart(fig, use_container_width=True)

# Download forecast
st.download_button("⬇️ Download Forecast CSV", forecast_df_all.to_csv(index=False), file_name=f"forecast_{selected_district}_2025_2035.csv")

st.markdown('</div>', unsafe_allow_html=True)

# ======= Footer =======
st.markdown("""
---
<center><small>Developed by AI for Climate-Aware Agriculture | © 2025</small></center>
""", unsafe_allow_html=True)