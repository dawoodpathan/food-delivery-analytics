"""
Food Delivery Operations - Data Hygiene, Feature Engineering & ML Pipeline
Author: Principal Data Analyst & ML Engineer
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in kilometers between two coordinate pairs."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0 * c

def parse_time_to_minutes(t_series):
    """Convert HH:MM:SS or HH:MM time strings into total minutes since midnight."""
    def convert(val):
        if pd.isna(val) or val == 'NaN':
            return np.nan
        parts = str(val).strip().split(':')
        if len(parts) >= 2:
            try:
                return int(parts[0]) * 60 + int(parts[1])
            except (ValueError, TypeError):
                return np.nan
        return np.nan
    return t_series.apply(convert)

def clean_and_prepare_data(csv_path='train.csv'):
    """
    Tier 1 Data Hygiene & Feature Engineering Pipeline:
    - Normalizes string encodings and handles whitespace
    - Replaces 'NaN' string literals with true NaN
    - Resolves coordinate anomalies (negative signs, 0 lat/long)
    - Computes physical Haversine delivery distance in km
    - Computes dispatch/prep time in minutes handling midnight rollover
    - Extracts temporal cohorts (hour, day of week, weekend indicator)
    - Cleans continuous delivery time and generates binary SLA breach target
    - Clips anomalous ratings and driver ages
    """
    print(f"[1/4] Loading raw dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    initial_shape = df.shape
    
    # 1. Clean whitespace and NaN string literals
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'str':
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'NaN': np.nan, 'nan': np.nan, '': np.nan, 'None': np.nan})

    # 2. Clean Target: Time_taken(min)
    df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min)', '', regex=False).str.strip()
    df['Time_taken_min'] = pd.to_numeric(df['Time_taken(min)'], errors='coerce')
    df = df.dropna(subset=['Time_taken_min'])
    df['Time_taken_min'] = df['Time_taken_min'].astype(int)

    # Define Primary Target: Delay_Status (SLA Breach if delivery time > 30 minutes)
    df['Delay_Status'] = (df['Time_taken_min'] > 30).astype(int)

    # 3. Clean Delivery Person Age & Ratings (Clip to realistic operational boundaries)
    df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce').clip(18, 65)
    df['Delivery_person_Age'] = df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].median())

    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce').clip(1.0, 5.0)
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median())

    # 4. Clean Weather conditions & Traffic
    df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '', regex=False).fillna('Sunny')
    df['Road_traffic_density'] = df['Road_traffic_density'].fillna('Medium')
    df['City'] = df['City'].fillna('Metropolitian')
    df['Festival'] = df['Festival'].fillna('No')
    df['Type_of_order'] = df['Type_of_order'].fillna('Meal')
    df['Type_of_vehicle'] = df['Type_of_vehicle'].fillna('motorcycle')
    df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce').fillna(1).clip(0, 3).astype(int)

    # 5. Geolocation Hygiene & Distance calculation
    # Negative coordinate correction (Indian cities are strictly Northern/Eastern hemisphere)
    lat1 = np.abs(pd.to_numeric(df['Restaurant_latitude'], errors='coerce'))
    lon1 = np.abs(pd.to_numeric(df['Restaurant_longitude'], errors='coerce'))
    lat2 = np.abs(pd.to_numeric(df['Delivery_location_latitude'], errors='coerce'))
    lon2 = np.abs(pd.to_numeric(df['Delivery_location_longitude'], errors='coerce'))

    calculated_dist = haversine_distance(lat1, lon1, lat2, lon2)
    # Filter 0,0 anomalies or impossible delivery distances (> 50 km in intra-city on-demand delivery)
    invalid_coords = (lat1 < 5) | (lon1 < 50) | (lat2 < 5) | (lon2 < 50) | (calculated_dist > 50) | (calculated_dist < 0.1)
    median_city_dist = calculated_dist[~invalid_coords].median()
    calculated_dist[invalid_coords] = median_city_dist
    df['Distance_km'] = np.round(calculated_dist, 2)

    # 6. Temporal Hygiene & Prep Time
    t_ordered = parse_time_to_minutes(df['Time_Orderd'])
    t_picked = parse_time_to_minutes(df['Time_Order_picked'])
    prep_time = t_picked - t_ordered
    # Handle midnight rollover
    prep_time = np.where(prep_time < 0, prep_time + 1440, prep_time)
    # Clip unreasonable prep times (e.g. > 60 min or <= 0 min) to median (10.0 min)
    prep_time = np.where((prep_time <= 0) | (prep_time > 60) | np.isnan(prep_time), 10.0, prep_time)
    df['Prep_Time_min'] = prep_time

    # Order Hour
    order_hour = (t_ordered // 60).fillna((t_picked // 60) - 0.2)
    order_hour = order_hour.fillna(19.0).astype(int) % 24
    df['Order_Hour'] = order_hour

    # Day of Week & Weekend
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
    df['Day_of_Week'] = df['Order_Date'].dt.day_name().fillna('Unknown')
    df['Is_Weekend'] = df['Order_Date'].dt.weekday.isin([5, 6]).astype(int)

    # City Code & Restaurant Entity
    df['City_Code'] = df['Delivery_person_ID'].str[:4]
    df['Restaurant_ID'] = df['Delivery_person_ID'].str.extract(r'([A-Z]+RES\d+)')[0].fillna('UNKNOWN_RES')

    # Transit Speed Proxy (km / hour)
    transit_time_hours = np.maximum(df['Time_taken_min'] - df['Prep_Time_min'], 5) / 60.0
    df['Speed_kmh'] = np.round(df['Distance_km'] / transit_time_hours, 2)
    df['Speed_kmh'] = df['Speed_kmh'].clip(5, 75)

    print(f"[OK] Data hygiene completed: Initial {initial_shape} -> Cleaned {df.shape}")
    return df

def generate_entity_aggregations(df):
    """
    Tier 1 Entity-Level Aggregation:
    - Delivery Partner Scorecard
    - Restaurant Kitchen & Dispatch Efficiency Index
    - City / Regional Logistics Hub Metrics
    """
    print("[2/4] Aggregating data at entity levels...")
    
    # 1. Delivery Partner Scorecard
    driver_stats = df.groupby('Delivery_person_ID').agg(
        Total_Deliveries=('ID', 'count'),
        Avg_Rating=('Delivery_person_Ratings', 'mean'),
        Avg_Delivery_Time=('Time_taken_min', 'mean'),
        SLA_Breach_Rate=('Delay_Status', 'mean'),
        Avg_Distance=('Distance_km', 'mean'),
        Most_Used_Vehicle=('Type_of_vehicle', lambda x: x.mode().iloc[0] if not x.empty else 'motorcycle')
    ).reset_index()
    driver_stats['SLA_Compliance_Rate'] = (1 - driver_stats['SLA_Breach_Rate']) * 100
    driver_stats['Avg_Delivery_Time'] = driver_stats['Avg_Delivery_Time'].round(1)
    driver_stats['Avg_Rating'] = driver_stats['Avg_Rating'].round(2)
    driver_stats['Avg_Distance'] = driver_stats['Avg_Distance'].round(2)

    # 2. Restaurant Dispatch Metrics
    restaurant_stats = df.groupby('Restaurant_ID').agg(
        Total_Orders=('ID', 'count'),
        Avg_Prep_Time=('Prep_Time_min', 'mean'),
        Avg_Delivery_Time=('Time_taken_min', 'mean'),
        Delay_Rate=('Delay_Status', 'mean'),
        City=('City_Code', 'first')
    ).reset_index()
    restaurant_stats['Avg_Prep_Time'] = restaurant_stats['Avg_Prep_Time'].round(1)
    restaurant_stats['Avg_Delivery_Time'] = restaurant_stats['Avg_Delivery_Time'].round(1)
    restaurant_stats['Delay_Rate_Pct'] = (restaurant_stats['Delay_Rate'] * 100).round(1)

    # 3. City Logistics Hub Metrics
    city_stats = df.groupby('City_Code').agg(
        Total_Volume=('ID', 'count'),
        Avg_Delivery_Time=('Time_taken_min', 'mean'),
        Delay_Rate=('Delay_Status', 'mean'),
        Avg_Distance=('Distance_km', 'mean'),
        Active_Drivers=('Delivery_person_ID', 'nunique')
    ).reset_index()
    city_stats['SLA_Compliance_Rate'] = ((1 - city_stats['Delay_Rate']) * 100).round(1)
    city_stats['Avg_Delivery_Time'] = city_stats['Avg_Delivery_Time'].round(1)
    city_stats['Avg_Distance'] = city_stats['Avg_Distance'].round(2)

    return driver_stats, restaurant_stats, city_stats

def build_and_evaluate_models(df, output_dir='models'):
    """
    Tier 3 Predictive Modeling with Strict Leakage Protection:
    - Removes raw IDs ('ID', 'Delivery_person_ID', 'Restaurant_ID')
    - Removes direct target derivatives ('Time_taken(min)', 'Time_taken_min', 'Speed_kmh')
    - Stratified 80/20 train/test split
    - Fits Logistic Regression (Baseline) & Random Forest (Champion)
    - Computes all operational metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
    - Extracts feature importance and decision thresholds
    """
    print("[3/4] Initiating ML training with leakage prevention...")
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = [
        'Delivery_person_Age', 'Delivery_person_Ratings', 'Distance_km',
        'Prep_Time_min', 'Vehicle_condition', 'multiple_deliveries',
        'Order_Hour', 'Weatherconditions', 'Road_traffic_density',
        'Type_of_vehicle', 'Type_of_order', 'Festival', 'City', 'Is_Weekend'
    ]

    num_cols = [
        'Delivery_person_Age', 'Delivery_person_Ratings', 'Distance_km',
        'Prep_Time_min', 'Vehicle_condition', 'multiple_deliveries',
        'Order_Hour', 'Is_Weekend'
    ]
    cat_cols = [
        'Weatherconditions', 'Road_traffic_density',
        'Type_of_vehicle', 'Type_of_order', 'Festival', 'City'
    ]

    X = df[feature_cols]
    y = df['Delay_Status']

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scikit-learn Pipeline Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    # 1. Baseline Model: Logistic Regression
    lr_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]

    lr_metrics = {
        'model_name': 'Logistic Regression (Baseline)',
        'accuracy': float(accuracy_score(y_test, lr_preds)),
        'precision': float(precision_score(y_test, lr_preds)),
        'recall': float(recall_score(y_test, lr_preds)),
        'f1': float(f1_score(y_test, lr_preds)),
        'roc_auc': float(roc_auc_score(y_test, lr_probs)),
        'confusion_matrix': confusion_matrix(y_test, lr_preds).tolist()
    }

    # 2. Production Model: Random Forest Classifier
    rf_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_split=8,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]

    rf_metrics = {
        'model_name': 'Random Forest Classifier (Production Champion)',
        'accuracy': float(accuracy_score(y_test, rf_preds)),
        'precision': float(precision_score(y_test, rf_preds)),
        'recall': float(recall_score(y_test, rf_preds)),
        'f1': float(f1_score(y_test, rf_preds)),
        'roc_auc': float(roc_auc_score(y_test, rf_probs)),
        'confusion_matrix': confusion_matrix(y_test, rf_preds).tolist()
    }

    # ROC and PR curves for champion model
    fpr, tpr, roc_thresh = roc_curve(y_test, rf_probs)
    precisions, recalls, pr_thresh = precision_recall_curve(y_test, rf_probs)

    # Feature importances
    fitted_prep = rf_pipeline.named_steps['prep']
    cat_feature_names = fitted_prep.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names
    importances = rf_pipeline.named_steps['clf'].feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n--- Model Benchmark Comparison ---")
    print(f"Logistic Regression -> Acc: {lr_metrics['accuracy']:.4f}, Prec: {lr_metrics['precision']:.4f}, Rec: {lr_metrics['recall']:.4f}, AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"Random Forest       -> Acc: {rf_metrics['accuracy']:.4f}, Prec: {rf_metrics['precision']:.4f}, Rec: {rf_metrics['recall']:.4f}, AUC: {rf_metrics['roc_auc']:.4f}")

    # Serialize artifacts
    joblib.dump(rf_pipeline, os.path.join(output_dir, 'champion_rf_pipeline.pkl'))
    joblib.dump(lr_pipeline, os.path.join(output_dir, 'baseline_lr_pipeline.pkl'))
    
    # Save evaluation summary metadata
    eval_artifacts = {
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'feature_cols': feature_cols,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'feat_imp': feat_imp_df.to_dict(orient='records'),
        'roc_data': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'pr_data': {'precision': precisions.tolist(), 'recall': recalls.tolist()},
        'test_actual': y_test.tolist(),
        'test_probs': rf_probs.tolist()
    }
    joblib.dump(eval_artifacts, os.path.join(output_dir, 'model_evaluation_artifacts.pkl'))
    print(f"[OK] Artifacts saved to '{output_dir}/'")

    return rf_pipeline, lr_pipeline, eval_artifacts

def run_full_pipeline():
    """Execute complete end-to-end data and modeling pipeline."""
    df_clean = clean_and_prepare_data('train.csv')
    
    # Save cleaned data
    clean_csv_path = 'cleaned_food_delivery.csv'
    df_clean.to_csv(clean_csv_path, index=False)
    print(f"[OK] Clean dataset saved to '{clean_csv_path}' ({len(df_clean)} records)")

    # Generate and save entity aggregations
    driver_stats, restaurant_stats, city_stats = generate_entity_aggregations(df_clean)
    driver_stats.to_csv('entity_driver_scorecard.csv', index=False)
    restaurant_stats.to_csv('entity_restaurant_metrics.csv', index=False)
    city_stats.to_csv('entity_city_hub_metrics.csv', index=False)
    print("[OK] Entity-level datasets saved (Drivers, Restaurants, City Hubs)")

    # Build and evaluate ML models
    build_and_evaluate_models(df_clean)
    print("\n[4/4] Pipeline execution complete! Production ready.")

if __name__ == '__main__':
    run_full_pipeline()
