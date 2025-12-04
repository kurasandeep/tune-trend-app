import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time, date
# Libraries for Data Acquisition and Modeling
from meteostat import Point, Daily
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# Spotify API Library and its specific exception
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException 
import time as time_module 

# --- 0. CONFIGURATION AND CONSTANTS ---

# Spotify Chart Playlist ID (Changed to RapCaviar: one of Spotify's largest and most stable public playlists)
# Old ID (Today's Top Hits): '37i9dQZEVXbJvfa0FSMvLq'
# New ID (RapCaviar): '37i9dQZF1DX0XUfTFmZEEK'
SPOTIFY_PLAYLIST_ID = '37i9dQZF1DX0XUfTFmZEEK' 
MAX_RETRIES = 3
MAX_POPULARITY = 100

# Define mapping for selected cities to their coordinates (Lat, Lon, Elevation, Name)
CITY_MAP = {
    "New York": (40.71, -74.01, 10, "New York City"),
    "London": (51.5, -0.12, 25, "London, UK"),
    "Sydney": (33.86, 151.2, 39, "Sydney, Australia"),
    "Tokyo": (35.68, 139.75, 40, "Tokyo, Japan"),
    "Miami": (25.76, -80.19, 2, "Miami, USA"),
    "Berlin": (52.52, 13.4, 34, "Berlin, Germany"),
}
DEFAULT_CITY = "New York"
DAYS_TO_ANALYZE = 180 
NUM_TRACKS = 5 
END_DATE = datetime.now().date() 
START_DATE = END_DATE - timedelta(days=DAYS_TO_ANALYZE)

# Features used by the Random Forest Model
FEATURES = [
    'popularity_lag_1', 'energy', 'valence', 'tempo', 'danceability', 
    'tavg', 'prcp', 'daylight_hours', 
    'month_sin', 'month_cos', 
    'is_weekend'
]

# --- SPOTIFY AUTHENTICATION ---

@st.cache_resource(show_spinner="Authenticating with Spotify...")
def authenticate_spotify():
    """Authenticates using credentials from Streamlit secrets."""
    try:
        # Securely read credentials from .streamlit/secrets.toml
        cid = st.secrets["SPOTIFY_CLIENT_ID"]
        secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
    except KeyError:
        st.error("Spotify credentials not found in Streamlit secrets. Please configure .streamlit/secrets.toml.")
        return None
    except st.errors.UnhashableTypeNameError:
        st.error("Error reading Spotify secrets. Ensure your keys are correctly formatted in secrets.toml.")
        return None

    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# --- 1. DATA ACQUISITION: SPOTIFY (Real-Time Data Fetching) ---

@st.cache_data(show_spinner="1. Fetching Spotify and Weather Data...")
def get_spotify_chart_data(_sp, start_date, end_date, num_days, num_tracks): # Parameter renamed to _sp for caching stability
    """
    Fetches daily top tracks' audio features and simulates a daily popularity score
    based on the real tracks' features.
    """
    if _sp is None: 
        return pd.DataFrame()

    track_data = []
    track_ids = []
    
    # 1. Fetch Top Tracks and Features TODAY
    try:
        # Robust API call: using the confirmed stable ID
        results = _sp.playlist_tracks(SPOTIFY_PLAYLIST_ID, limit=num_tracks) 
    except SpotifyException as e:
        # Handle the 404/Resource Not Found error specifically
        st.error(f"Error fetching playlist tracks: {e}. Please ensure the SPOTIFY_PLAYLIST_ID is public and correct.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during Spotify API call: {e}")
        return pd.DataFrame()

    # Continue with processing fetched data
    for item in results['items']:
        track = item['track']
        if track and track['id'] and track['id'] not in track_ids:
            track_ids.append(track['id'])
            
            base_pop = track.get('popularity', 50) 
            
            # Fetch audio features for the track
            try:
                features = _sp.audio_features(track['id'])[0] 
            except Exception:
                features = None
            
            if features:
                track_data.append({
                    'id': track['id'],
                    'name': track['name'],
                    'base_popularity': base_pop,
                    'energy': features.get('energy', 0.5),
                    'valence': features.get('valence', 0.5),
                    'tempo': features.get('tempo', 120),
                    'danceability': features.get('danceability', 0.5),
                })

    if not track_data:
        st.warning("No track data found. API credentials might be invalid or the playlist is empty.")
        return pd.DataFrame()

    # 2. Simulate Historical Daily Popularity using fixed features
    date_range = [end_date - timedelta(days=d) for d in range(num_days)]
    historical_data = []

    for track in track_data:
        for single_date in date_range:
            # Seasonal/Random Popularity Component 
            # Sine wave component to introduce seasonality bias 

            seasonal_factor = 10 * np.sin(2 * np.pi * single_date.timetuple().tm_yday / 365)
            
            # Combine base popularity with seasonal noise
            popularity = np.clip(
                track['base_popularity'] + seasonal_factor + np.random.normal(0, 5), 
                10, 
                MAX_POPULARITY
            )

            historical_data.append({
                'date': single_date,
                'track_id': track['id'],
                'popularity': int(popularity),
                'energy': track['energy'],
                'valence': track['valence'],
                'tempo': track['tempo'],
                'danceability': track['danceability'],
            })
    
    st.success(f"Successfully fetched {len(track_data)} tracks and simulated {len(historical_data)} days of data.")
    return pd.DataFrame(historical_data)


# --- 2. DATA ACQUISITION: WEATHER & INTEGRATION ---

@st.cache_data(show_spinner="1. Fetching Spotify and Weather Data...")
def get_integrated_data(_sp, lat, lon, elevation, location_name, start_date, end_date): # Parameter renamed to _sp for caching stability
    """Acquires, simulates, and integrates all project data."""
    
    # Internal function to fetch weather data from Meteostat
    def get_meteostat_weather_data(lat, lon, start, end, location_name, elevation):
        
        # Convert input datetime.date objects to datetime.datetime 
        start_dt = datetime.combine(start, time(0, 0))
        end_dt = datetime.combine(end, time(0, 0))
        
        location = Point(lat, lon, elevation)
        data = Daily(location, start_dt, end_dt) 
        weather_df = data.fetch()
        weather_df['location'] = location_name
        
        weather_df = weather_df.reset_index().rename(columns={'time': 'date'})
        weather_df = weather_df[['date', 'location', 'tavg', 'prcp', 'tsun']].copy()
        weather_df['daylight_hours'] = weather_df['tsun'].fillna(0) / 60 
        weather_df.drop(columns=['tsun'], inplace=True)
        return weather_df.dropna(subset=['tavg'])

    days_to_analyze = (end_date - start_date).days + 1
    
    # 1. Get Spotify Data using the Spotipy client
    music_df = get_spotify_chart_data(_sp, start_date, end_date, days_to_analyze, NUM_TRACKS) 
    
    if music_df.empty:
        st.warning("Spotify data is empty. Skipping model training and forecast.")
        return pd.DataFrame()

    # 2. Get Meteostat Weather Data (Dynamic Location)
    weather_df = get_meteostat_weather_data(lat, lon, start_date, end_date, location_name, elevation)

    # 3. Merge and Feature Engineering
    
    music_df['date'] = pd.to_datetime(music_df['date']).dt.normalize()
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.normalize()
    
    master_df = pd.merge(music_df, weather_df, on='date', how='inner') 

    # Time Features 
    master_df['month'] = master_df['date'].dt.month
    master_df['day_of_week'] = master_df['date'].dt.dayofweek
    master_df['is_weekend'] = master_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Cyclical Features
    master_df['month_sin'] = np.sin(2 * np.pi * master_df['month'] / 12)
    master_df['month_cos'] = np.cos(2 * np.pi * master_df['month'] / 12)
    
    # Lagged Popularity
    master_df = master_df.sort_values(by=['track_id', 'date']).reset_index(drop=True)
    master_df['popularity_lag_1'] = master_df.groupby('track_id')['popularity'].shift(1)
    master_df.dropna(subset=['popularity_lag_1'], inplace=True)
    
    return master_df

# --- 3. MODEL TRAINING & FORECASTING ---

@st.cache_resource(show_spinner="2. Training Random Forest Regressor...")
def train_model(data_df):
    """Trains the Random Forest Regressor and extracts validation data."""
    
    X = data_df[FEATURES]
    y = data_df['popularity']

    # Chronological Split (80% Train, 20% Test)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # Train Model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict and Validate
    y_pred = model.predict(X_test)
    
    validation_df = pd.DataFrame({
        'date': data_df['date'].dt.date.tail(len(y_test)), 
        'actual_popularity': y_test,
        'predicted_popularity': y_pred
    })
    
    return model, validation_df

@st.cache_data(show_spinner="3. Generating 30-Day Forecast...")
def generate_forecast(_model, data_df, days=30): 
    """Generates a forward 30-day forecast based on the trained model."""
    
    last_date = data_df['date'].max() 
    future_dates = [last_date.date() + timedelta(days=d) for d in range(1, days + 1)] 
    forecast_df = pd.DataFrame({'date': future_dates})

    # Prepare stats for simulating future weather/lag
    seasonal_stats = data_df[['tavg', 'prcp', 'daylight_hours']].mean() 

    # Simulate Future Features
    forecast_df['tavg'] = seasonal_stats['tavg'] + np.random.normal(0, 3, days)
    forecast_df['prcp'] = seasonal_stats['prcp'] * np.random.uniform(0.5, 1.5, days)
    forecast_df['daylight_hours'] = seasonal_stats['daylight_hours'] + np.random.normal(0, 1, days)

    # Feature Engineering for Forecast
    forecast_df['month'] = forecast_df['date'].apply(lambda x: x.month)
    forecast_df['is_weekend'] = forecast_df['date'].apply(lambda x: x.weekday() >= 5) 
    forecast_df['month_sin'] = np.sin(2 * np.pi * forecast_df['month'] / 12)
    forecast_df['month_cos'] = np.cos(2 * np.pi * forecast_df['month'] / 12)

    # Lagged Popularity (Using the mean of the training set)
    last_lag_value = data_df['popularity_lag_1'].mean()
    forecast_df['popularity_lag_1'] = last_lag_value

    # Song Attributes (Average values from the training set)
    for feature in ['energy', 'valence', 'tempo', 'danceability']:
        forecast_df[feature] = data_df[feature].mean()

    # Make Forecast
    X_forecast = forecast_df[FEATURES]
    forecast_predictions = _model.predict(X_forecast) 

    forecast_df['predicted_popularity'] = np.round(forecast_predictions).astype(int)
    return forecast_df[['date', 'predicted_popularity', 'tavg', 'daylight_hours', 'month']]


# --- 4. STREAMLIT APPLICATION UI & VISUALIZATION ---

st.set_page_config(layout="wide", page_title="Tune Trend: Spotify Forecasting 🎶")
st.title("🎶 Tune Trend: Spotify Chart Trend Forecasting Dashboard")

# Authenticate Spotify at the start
sp_client = authenticate_spotify()

# --- UI INPUT (Sidebar) ---
st.sidebar.header("Location & Analysis Settings")

selected_city = st.sidebar.selectbox(
    "Select a City to Analyze:",
    options=list(CITY_MAP.keys()),
    index=list(CITY_MAP.keys()).index(DEFAULT_CITY)
)

# Resolve coordinates based on user selection
lat, lon, elevation, location_name = CITY_MAP[selected_city]

st.sidebar.markdown(f"""
    ---
    **Spotify Data Source:** Top {NUM_TRACKS} tracks from {SPOTIFY_PLAYLIST_ID}  
    **Weather Data Source:** {location_name}  
    **Training Period:** {DAYS_TO_ANALYZE} Days ({START_DATE} to {END_DATE})
""")
st.sidebar.info("The **Tune Trend** model predicts future song popularity by correlating real-time Spotify track features (like energy and danceability) with local weather and seasonal trends.")
# --- End UI Input ---

if sp_client is not None:
    # Execution Order: Load Data -> Train Model -> Generate Forecast
    master_df = get_integrated_data(sp_client, lat, lon, elevation, location_name, START_DATE, END_DATE) 

    if master_df.empty:
        # If master_df is empty, it means Spotify data retrieval failed (handled in previous functions)
        st.error("Model training aborted. The application could not retrieve or process sufficient Spotify/Weather data. Check your logs and the Playlist ID.")
    else:
        model, validation_df = train_model(master_df)
        forecast_df = generate_forecast(model, master_df) 

        # Calculate final R2 metric for display
        r2 = r2_score(validation_df['actual_popularity'], validation_df['predicted_popularity'])

        # --- Dashboard Header ---
        st.markdown(f"### Location: **{location_name}** | Model Accuracy (R² Score): **{r2:.4f}**")

        # --- Dashboard Sections (Visualization) ---

        st.header("1. Historical Model Validation (Actual vs. Predicted)")

        fig_validation = go.Figure()
        fig_validation.add_trace(go.Scatter(x=validation_df['date'], y=validation_df['actual_popularity'],
                                            mode='lines', name='Actual Popularity (Simulated)', line=dict(color='red', width=3)))
        fig_validation.add_trace(go.Scatter(x=validation_df['date'], y=validation_df['predicted_popularity'],
                                            mode='lines', name='Predicted Popularity', line=dict(color='blue', dash='dot', width=2)))

        fig_validation.update_layout(xaxis_title='Date', yaxis_title='Popularity Score', hovermode="x unified", height=400)
        st.plotly_chart(fig_validation, use_container_width=True)

        # ----------------------------------------------------
        ## 2. 30-Day Forward Trend Forecast
        # ----------------------------------------------------
        st.header("2. 30-Day Forward Trend Forecast")

        col1, col2 = st.columns(2)

        with col1:
            # Chart A: Predicted Popularity
            fig_forecast_pop = go.Figure()
            fig_forecast_pop.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_popularity'],
                                                   mode='lines+markers', name='Predicted Popularity', line=dict(color='red', width=3)))
            fig_forecast_pop.update_layout(title='Predicted Song Popularity Trend', xaxis_title='Date', yaxis_title='Popularity Score', hovermode="x unified", height=400)
            st.plotly_chart(fig_forecast_pop, use_container_width=True)

        with col2:
            # Chart B: Forecasted Weather Drivers
            fig_weather = go.Figure()
            fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['tavg'],
                                             mode='lines', name='Avg. Temp (°C)', yaxis='y1', line=dict(color='orange', width=2)))
            
            fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['daylight_hours'],
                                             mode='lines', name='Daylight Hours', yaxis='y2', line=dict(color='skyblue', dash='dot', width=2)))

            fig_weather.update_layout(title='Forecasted Weather Drivers', xaxis_title='Date',
                yaxis=dict(title='Avg. Temp (°C)', color='orange'),
                yaxis2=dict(title='Daylight Hours', overlaying='y', side='right', color='skyblue'),
                hovermode="x unified", height=400)
            st.plotly_chart(fig_weather, use_container_width=True)

        # ----------------------------------------------------
        ## 3. Seasonal Trends of Key Features
        # ----------------------------------------------------
        st.header("3. Seasonal Trends of Key Features")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Prepare data for seasonal visualization
        seasonal_df = master_df.groupby(master_df['date'].dt.month)[['valence', 'energy', 'tavg']].mean().reset_index()
        seasonal_df['Month'] = seasonal_df['date'].apply(lambda x: months[x - 1])

        # Plot Valence (Mood) and Energy vs. Month
        fig_seasonal = px.line(seasonal_df, x='Month', y=['valence', 'energy'], 
                               title=f'Average Song Valence (Mood) & Energy vs. Temperature in {selected_city}',
                               labels={'value': 'Score (0-1)', 'Month': 'Month'},
                               color_discrete_map={'valence': 'green', 'energy': 'blue'})

        # Add Temperature as a secondary axis for seasonal context
        fig_seasonal.add_trace(go.Scatter(x=seasonal_df['Month'], y=seasonal_df['tavg'], 
                                          name='Avg. Temp (°C)', yaxis='y2', mode='lines', 
                                          line=dict(color='red', dash='dash')))

        fig_seasonal.update_layout(
            yaxis=dict(title='Score (0-1)'),
            yaxis2=dict(title='Avg. Temp (°C)', overlaying='y', side='right', showgrid=False),
            hovermode="x unified"
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
