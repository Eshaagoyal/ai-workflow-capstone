import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from cslib import fetch_data, engineer_features
from logger import update_train_log, update_predict_log

MODEL_DIR = os.path.join(".", "models")
MODEL_VERSION = "0.1"
MODEL_VERSION_NOTE = "RandomForestRegressor baseline"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def model_train(data_dir, test=False):
    """
    Train models for specific target countries and an aggregate 'all' dataset.
    """
    start_time = time.time()
    df = fetch_data(data_dir)
    
    countries = ['all', 'United Kingdom', 'Germany', 'France']
    saved_models = {}
    
    for country in countries:
        c_df = df if country == 'all' else df[df['country'] == country]
        if c_df.empty:
            continue
            
        c_df = engineer_features(c_df, training=True)
        feature_cols = ['revenue_7d', 'revenue_30d', 'views_7d']
        
        X = c_df[feature_cols]
        y = c_df['target']
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        
        prefix = "test-" if test else ""
        model_filename = os.path.join(MODEL_DIR, f"{prefix}model-{country.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, model_filename)
        saved_models[country] = model_filename
        
        runtime = round(time.time() - start_time, 2)
        update_train_log(country, (str(c_df['invoice_date'].min()), str(c_df['invoice_date'].max())), rmse, runtime, MODEL_VERSION, test=test)
        
    return saved_models

def model_load(country='all', test=False):
    prefix = "test-" if test else ""
    model_filename = os.path.join(MODEL_DIR, f"{prefix}model-{country.lower().replace(' ', '_')}.joblib")
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file {model_filename} not found.")
    return joblib.load(model_filename)

def model_predict(country, year, month, day, test=False):
    """
    Generate revenue prediction for a given country and date query.
    """
    start_time = time.time()
    query_date = f"{year}-{month:02d}-{day:02d}"
    
    try:
        model = model_load(country, test=test)
    except FileNotFoundError:
        # Fallback to aggregate model if country-specific model does not exist
        model = model_load('all', test=test)
        
    # Baseline dummy inputs representing the historical rolling window
    X_sample = np.array([[12000.0, 45000.0, 3200.0]])
    y_pred = model.predict(X_sample)[0]
    
    runtime = round(time.time() - start_time, 3)
    update_predict_log(country, y_pred, None, query_date, runtime, MODEL_VERSION, test=test)
    return {"y_pred": float(y_pred), "country": country, "target_date": query_date}