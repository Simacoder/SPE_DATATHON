import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr

print("=== SPE DSEATS AFRICA DATATHON 2025 - XGBoost Well Classification ===")
print("Loading and preprocessing datasets...")

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_and_explore_data():
    """Load all datasets and perform initial exploration"""
    
    # Load datasets
    production_data = pd.read_csv('data/spe_africa_dseats_datathon_2025_wells_dataset.csv')
    classification_params = pd.read_csv('data/classification_parameters.csv')
    reservoir_info = pd.read_csv('data/reservoir_info.csv')
    
    print(f"Production data shape: {production_data.shape}")
    print(f"Classification parameters shape: {classification_params.shape}")
    print(f"Reservoir info shape: {reservoir_info.shape}")
    
    # Print column names to identify the correct well identifier
    print(f"\nProduction data columns: {production_data.columns.tolist()}")
    print(f"Classification parameters columns: {classification_params.columns.tolist()}")
    print(f"Reservoir info columns: {reservoir_info.columns.tolist()}")
    
    # Try to identify the well name column
    well_col = None
    for col in production_data.columns:
        if 'WELL' in col.upper() or 'NAME' in col.upper():
            well_col = col
            break
    
    if well_col:
        print(f"Well identifier column: {well_col}")
        print(f"Unique wells in production data: {production_data[well_col].nunique()}")
    else:
        print("Warning: Could not identify well name column")
    
    return production_data, classification_params, reservoir_info

# ============================================================================
# 2. DATA CLEANING AND PREPROCESSING
# ============================================================================

def clean_production_data(df):
    """Clean and preprocess production data"""
    
    print("\n=== DATA CLEANING ===")
    
    # Identify the well name column
    well_col = None
    for col in df.columns:
        if 'WELL' in col.upper() or 'NAME' in col.upper():
            well_col = col
            break
    
    if not well_col:
        # Use the first column as well identifier if no clear well name column
        well_col = df.columns[0]
        print(f"Using '{well_col}' as well identifier")
    
    # Identify date column
    date_col = None
    for col in df.columns:
        if 'DATE' in col.upper() or 'TIME' in col.upper():
            date_col = col
            break
    
    if date_col:
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        print(f"Converted date column: {date_col}")
    
    # Handle string columns that should be numeric
    numeric_columns = []
    for col in df.columns:
        if any(keyword in col.upper() for keyword in ['PRESSURE', 'PROD', 'CUMULATIVE', 'TEMPERATURE', 'CHOKE']):
            numeric_columns.append(col)
    
    print(f"Identified numeric columns: {numeric_columns}")
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, errors='coerce' will convert non-numeric to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check missing values
    print("Missing values after cleaning:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Fill missing values with forward fill method for time series (updated method)
    if date_col:
        df_cleaned = df.sort_values([well_col, date_col])
        df_cleaned = df_cleaned.groupby(well_col).ffill()
        df_cleaned = df_cleaned.groupby(well_col).bfill()
    else:
        df_cleaned = df.sort_values(well_col)
        df_cleaned = df_cleaned.groupby(well_col).ffill()
        df_cleaned = df_cleaned.groupby(well_col).bfill()
    
    # Add well column name to the dataframe for later use
    df_cleaned._well_col = well_col
    df_cleaned._date_col = date_col
    
    return df_cleaned

def clean_reservoir_data(df):
    """Clean reservoir information data"""
    
    # Convert string columns to numeric
    numeric_columns = []
    for col in df.columns:
        if any(keyword in col.upper() for keyword in ['PRESSURE', 'RATIO', 'POINT']):
            numeric_columns.append(col)
    
    print(f"Reservoir numeric columns: {numeric_columns}")
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def engineer_production_features(df):
    """Engineer features from production data"""
    
    print("\n=== FEATURE ENGINEERING ===")
    
    well_col = df._well_col
    date_col = df._date_col
    
    # Sort by well and date if date column exists
    if date_col:
        df = df.sort_values([well_col, date_col])
    else:
        df = df.sort_values(well_col)
    
    well_features = []
    
    for well in df[well_col].unique():
        if pd.isna(well):
            continue
            
        well_data = df[df[well_col] == well].copy()
        
        if len(well_data) == 0:
            continue
            
        # Initialize features dictionary
        features = {'WELL_NAME': well, 'total_records': len(well_data)}
        
        # Time-based features if date column exists
        if date_col and not well_data[date_col].isna().all():
            production_days = (well_data[date_col].max() - well_data[date_col].min()).days
            features['production_days'] = production_days
        
        # Process all numeric columns
        numeric_cols = well_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in well_data.columns and not well_data[col].isna().all():
                col_clean = col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
                
                # Basic statistics
                features[f'avg_{col_clean}'] = well_data[col].mean()
                features[f'max_{col_clean}'] = well_data[col].max()
                features[f'min_{col_clean}'] = well_data[col].min()
                features[f'std_{col_clean}'] = well_data[col].std()
                features[f'sum_{col_clean}'] = well_data[col].sum()
                
                # Trend analysis (if enough data points)
                if len(well_data) > 1:
                    first_val = well_data[col].iloc[0]
                    last_val = well_data[col].iloc[-1]
                    if not pd.isna(first_val) and not pd.isna(last_val):
                        features[f'{col_clean}_change'] = last_val - first_val
                        features[f'{col_clean}_change_rate'] = (last_val - first_val) / len(well_data)
        
        # Calculate derived ratios for cumulative production columns
        cum_oil_col = None
        cum_water_col = None
        cum_gas_col = None
        
        for col in well_data.columns:
            if 'CUMULATIVE' in col.upper() and 'OIL' in col.upper():
                cum_oil_col = col
            elif 'CUMULATIVE' in col.upper() and 'WATER' in col.upper():
                cum_water_col = col
            elif 'CUMULATIVE' in col.upper() and 'GAS' in col.upper():
                cum_gas_col = col
        
        # Calculate ratios if columns exist
        if cum_oil_col and cum_water_col:
            oil_prod = well_data[cum_oil_col].iloc[-1] if len(well_data) > 0 else 0
            water_prod = well_data[cum_water_col].iloc[-1] if len(well_data) > 0 else 0
            
            if oil_prod > 0:
                features['water_oil_ratio'] = water_prod / oil_prod
            
            total_liquid = oil_prod + water_prod
            if total_liquid > 0:
                features['water_cut'] = water_prod / total_liquid
        
        if cum_oil_col and cum_gas_col:
            oil_prod = well_data[cum_oil_col].iloc[-1] if len(well_data) > 0 else 0
            gas_prod = well_data[cum_gas_col].iloc[-1] if len(well_data) > 0 else 0
            
            if oil_prod > 0:
                features['gas_oil_ratio'] = gas_prod / oil_prod
        
        well_features.append(features)
    
    feature_df = pd.DataFrame(well_features)
    
    # Fill NaN values with 0 for numerical features
    feature_df = feature_df.fillna(0)
    
    # Replace infinite values with 0
    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    
    print(f"Engineered features shape: {feature_df.shape}")
    print(f"Number of features: {len(feature_df.columns)}")
    
    return feature_df

# ============================================================================
# 4. STATISTICAL ANALYSIS
# ============================================================================

def perform_statistical_analysis(features_df, target_df):
    """Perform statistical analysis on features"""
    
    print("\n=== STATISTICAL ANALYSIS ===")
    
    # Basic statistics for key features
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    print(f"Number of numeric features: {len(numeric_features)}")
    
    if len(numeric_features) > 0:
        print("Basic statistics for first 10 numeric features:")
        for i, feature in enumerate(numeric_features[:10]):
            print(f"\n{feature}:")
            print(f"  Mean: {features_df[feature].mean():.2f}")
            print(f"  Std: {features_df[feature].std():.2f}")
            print(f"  Min: {features_df[feature].min():.2f}")
            print(f"  Max: {features_df[feature].max():.2f}")
    
    # Correlation analysis
    if len(numeric_features) > 1:
        print("\nCorrelation analysis (top 10 correlations):")
        corr_matrix = features_df[numeric_features].corr()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find the top correlations
        correlations = []
        for col in upper.columns:
            for idx in upper.index:
                if not pd.isna(upper.loc[idx, col]):
                    correlations.append((abs(upper.loc[idx, col]), idx, col, upper.loc[idx, col]))
        
        correlations.sort(reverse=True)
        for i, (abs_corr, feat1, feat2, corr) in enumerate(correlations[:10]):
            print(f"  {feat1} vs {feat2}: {corr:.3f}")
        
        return corr_matrix
    
    return None

# ============================================================================
# 5. TARGET VARIABLE CREATION
# ============================================================================

def create_target_variable(classification_params, features_df):
    """Create target variable for classification"""
    
    print("\n=== TARGET VARIABLE CREATION ===")
    
    # Print available columns in classification_params
    print(f"Available columns in classification_params: {classification_params.columns.tolist()}")
    
    # Find well identifier column in classification_params
    well_id_col = None
    for col in classification_params.columns:
        if 'WELL' in col.upper() or 'NAME' in col.upper() or 'RESERVOIR' in col.upper():
            well_id_col = col
            break
    
    if not well_id_col:
        well_id_col = classification_params.columns[0]
    
    print(f"Using '{well_id_col}' as well identifier in classification_params")
    
    # Create target variable
    target_df = classification_params.copy()
    
    # Try to find a suitable target column
    target_col = None
    for col in target_df.columns:
        if any(keyword in col.upper() for keyword in ['TYPE', 'CLASS', 'CATEGORY']):
            target_col = col
            break
    
    if target_col:
        target_df['target_class'] = target_df[target_col].astype(str)
        print(f"Using '{target_col}' as target variable")
    else:
        # Create synthetic target based on available columns
        non_id_cols = [col for col in target_df.columns if col != well_id_col]
        if len(non_id_cols) >= 2:
            target_df['target_class'] = target_df[non_id_cols[0]].astype(str) + '_' + target_df[non_id_cols[1]].astype(str)
        else:
            target_df['target_class'] = target_df[non_id_cols[0]].astype(str) if non_id_cols else 'default'
        print(f"Created synthetic target from: {non_id_cols[:2]}")
    
    # Clean target classes
    target_df['target_class'] = target_df['target_class'].str.strip()
    target_df = target_df.dropna(subset=['target_class'])
    
    # Encode target variable
    le = LabelEncoder()
    target_df['target_encoded'] = le.fit_transform(target_df['target_class'])
    
    print(f"Target classes: {target_df['target_class'].unique()}")
    print(f"Target distribution:")
    print(target_df['target_class'].value_counts())
    
    # Merge with features
    final_df = features_df.merge(target_df[[well_id_col, 'target_class', 'target_encoded']], 
                                left_on='WELL_NAME', right_on=well_id_col, how='inner')
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Wells with both features and targets: {len(final_df)}")
    
    return final_df, le

# ============================================================================
# 6. MODEL TRAINING AND EVALUATION
# ============================================================================

def train_xgboost_model(df, target_col='target_encoded'):
    """Train XGBoost model"""
    
    print("\n=== MODEL TRAINING ===")
    
    # Prepare features and target
    exclude_cols = ['WELL_NAME', 'target_class', 'target_encoded']
    # Also exclude any column that might be the well identifier from classification_params
    for col in df.columns:
        if 'Reservoir' in col or 'Well' in col:
            exclude_cols.append(col)
    
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_columns]
    y = df[target_col]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Check if we have enough samples for each class
    class_counts = pd.Series(y).value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    # Remove constant features
    constant_features = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)
        feature_columns = [col for col in feature_columns if col not in constant_features]
    
    # Split the data
    test_size = min(0.3, 0.8)  # Adjust test size for small datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=50,  # Reduced for small datasets
        max_depth=4,      # Reduced to prevent overfitting
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = xgb_model.predict(X_test_scaled)
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    return xgb_model, scaler, feature_columns, feature_importance

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for XGBoost"""
    
    print("\n=== HYPERPARAMETER TUNING ===")
    
    param_grid = {
        'n_estimators': [30, 50, 100],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
    
    # Use smaller CV for small datasets
    cv_folds = min(3, len(X_train) // 5)
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# ============================================================================
# 7. PREDICTION AND OUTPUT
# ============================================================================

def generate_predictions(model, scaler, feature_columns, all_wells_df, le):
    """Generate predictions for all wells"""
    
    print("\n=== GENERATING PREDICTIONS ===")
    
    # Prepare features for all wells
    X_all = all_wells_df[feature_columns]
    X_all_scaled = scaler.transform(X_all)
    
    # Make predictions
    predictions = model.predict(X_all_scaled)
    prediction_proba = model.predict_proba(X_all_scaled)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'WELL_NAME': all_wells_df['WELL_NAME'],
        'PREDICTED_CLASS_ENCODED': predictions,
        'PREDICTED_CLASS': le.inverse_transform(predictions)
    })
    
    # Add prediction probabilities
    for i, class_name in enumerate(le.classes_):
        output_df[f'PROBABILITY_{class_name}'] = prediction_proba[:, i]
    
    return output_df

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    try:
        # Load data
        production_data, classification_params, reservoir_info = load_and_explore_data()
        
        # Clean data
        production_cleaned = clean_production_data(production_data)
        reservoir_cleaned = clean_reservoir_data(reservoir_info)
        
        # Engineer features
        features_df = engineer_production_features(production_cleaned)
        
        # Check if we have enough data to proceed
        if len(features_df) == 0:
            print("Error: No features were generated. Please check your data.")
            return None, None
        
        # Statistical analysis
        corr_matrix = perform_statistical_analysis(features_df, classification_params)
        
        # Create target variable
        final_df, label_encoder = create_target_variable(classification_params, features_df)
        
        # Check if we have enough data for training
        if len(final_df) < 10:
            print("Error: Not enough data for training. Need at least 10 samples.")
            return None, None
        
        # Train model
        model, scaler, feature_columns, feature_importance = train_xgboost_model(final_df)
        
        # Generate predictions for all wells
        predictions_df = generate_predictions(model, scaler, feature_columns, final_df, label_encoder)
        
        # Save results
        predictions_df.to_csv('well_classification_predictions.csv', index=False)
        feature_importance.to_csv('feature_importance.csv', index=False)
        
        print("\n=== RESULTS SAVED ===")
        print("Files created:")
        print("- well_classification_predictions.csv: Final predictions")
        print("- feature_importance.csv: Feature importance scores")
        
        print("\nPrediction Summary:")
        print(predictions_df['PREDICTED_CLASS'].value_counts())
        
        return predictions_df, feature_importance
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Run the complete pipeline
if __name__ == "__main__":
    predictions, feature_importance = main()