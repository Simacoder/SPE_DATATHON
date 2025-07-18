import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr
import os

print("=== SPE DSEATS AFRICA DATATHON 2025 - Enhanced XGBoost Well Classification ===")
print("Loading and preprocessing datasets...")

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_and_explore_data():
    """Load all datasets and perform initial exploration"""
    
    try:
        # Load datasets with error handling
        file_paths = {
            'production': 'data/spe_africa_dseats_datathon_2025_wells_dataset.csv',
            'classification': 'data/classification_parameters.csv',
            'reservoir': 'data/reservoir_info.csv'
        }
        
        datasets = {}
        for name, path in file_paths.items():
            if os.path.exists(path):
                datasets[name] = pd.read_csv(path)
                print(f"✓ Loaded {name} data: {datasets[name].shape}")
            else:
                print(f"⚠ Warning: {path} not found. Skipping {name} data.")
                datasets[name] = None
        
        # If main production file doesn't exist, try alternative names
        if datasets['production'] is None:
            alternative_names = ['wells_dataset.csv', 'production_data.csv', 'wells_data.csv']
            for alt_name in alternative_names:
                if os.path.exists(alt_name):
                    datasets['production'] = pd.read_csv(alt_name)
                    print(f"✓ Found alternative production file: {alt_name}")
                    break
        
        production_data = datasets['production']
        classification_params = datasets['classification']
        reservoir_info = datasets['reservoir']
        
        if production_data is None:
            raise FileNotFoundError("Production data file not found. Please ensure the data file exists.")
        
        # Print detailed information about datasets
        print("\n=== DATASET OVERVIEW ===")
        print(f"Production data shape: {production_data.shape}")
        print(f"Production columns: {production_data.columns.tolist()}")
        
        if classification_params is not None:
            print(f"Classification parameters shape: {classification_params.shape}")
            print(f"Classification columns: {classification_params.columns.tolist()}")
        
        if reservoir_info is not None:
            print(f"Reservoir info shape: {reservoir_info.shape}")
            print(f"Reservoir columns: {reservoir_info.columns.tolist()}")
        
        # Identify well identifier column
        well_col = identify_well_column(production_data)
        if well_col:
            print(f"✓ Well identifier column: {well_col}")
            print(f"✓ Unique wells: {production_data[well_col].nunique()}")
            print(f"✓ Sample well names: {production_data[well_col].unique()[:5]}")
        
        return production_data, classification_params, reservoir_info
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def identify_well_column(df):
    """Identify the well name column in the dataframe"""
    well_keywords = ['WELL', 'NAME', 'ID', 'IDENTIFIER']
    
    for col in df.columns:
        if any(keyword in col.upper() for keyword in well_keywords):
            return col
    
    # If no clear well column found, return first column
    return df.columns[0] if len(df.columns) > 0 else None

# ============================================================================
# 2. ENHANCED DATA CLEANING AND PREPROCESSING
# ============================================================================

def clean_production_data(df):
    """Enhanced cleaning and preprocessing of production data"""
    
    print("\n=== ENHANCED DATA CLEANING ===")
    
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Identify key columns
    well_col = identify_well_column(df_clean)
    date_col = identify_date_column(df_clean)
    
    if not well_col:
        raise ValueError("Could not identify well column")
    
    print(f"✓ Well column: {well_col}")
    print(f"✓ Date column: {date_col if date_col else 'Not found'}")
    
    # Handle date column if exists
    if date_col:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        print(f"✓ Converted {date_col} to datetime")
        
        # Remove rows with invalid dates
        invalid_dates = df_clean[date_col].isna().sum()
        if invalid_dates > 0:
            print(f"⚠ Removed {invalid_dates} rows with invalid dates")
            df_clean = df_clean.dropna(subset=[date_col])
    
    # Identify and clean numeric columns
    numeric_columns = identify_numeric_columns(df_clean)
    print(f"✓ Identified {len(numeric_columns)} numeric columns")
    
    # Convert to numeric and handle outliers
    for col in numeric_columns:
        if col in df_clean.columns:
            # Convert to numeric
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Handle outliers using IQR method
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"⚠ Found {outliers} outliers in {col}")
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Handle missing values more intelligently
    print("✓ Handling missing values...")
    
    # Sort data for proper forward/backward filling
    sort_cols = [well_col]
    if date_col:
        sort_cols.append(date_col)
    
    df_clean = df_clean.sort_values(sort_cols)
    
    # Fill missing values by well
    for col in numeric_columns:
        if col in df_clean.columns:
            # Forward fill, then backward fill, then median fill
            df_clean[col] = df_clean.groupby(well_col)[col].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(x.median()))
    
    # Final check for missing values
    missing_summary = df_clean.isnull().sum()
    if missing_summary.sum() > 0:
        print("⚠ Remaining missing values:")
        print(missing_summary[missing_summary > 0])
    
    # Store metadata
    df_clean._well_col = well_col
    df_clean._date_col = date_col
    df_clean._numeric_cols = numeric_columns
    
    print(f"✓ Cleaned data shape: {df_clean.shape}")
    return df_clean

def identify_date_column(df):
    """Identify date column in dataframe"""
    date_keywords = ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']
    
    for col in df.columns:
        if any(keyword in col.upper() for keyword in date_keywords):
            return col
    
    return None

def identify_numeric_columns(df):
    """Identify columns that should be numeric"""
    numeric_keywords = [
        'PRESSURE', 'PROD', 'CUMULATIVE', 'TEMPERATURE', 'CHOKE', 
        'RATE', 'FLOW', 'VOLUME', 'DEPTH', 'PERCENT', 'RATIO'
    ]
    
    numeric_columns = []
    for col in df.columns:
        if any(keyword in col.upper() for keyword in numeric_keywords):
            numeric_columns.append(col)
    
    return numeric_columns

# ============================================================================
# 3. ENHANCED FEATURE ENGINEERING
# ============================================================================

def engineer_production_features(df):
    """Enhanced feature engineering from production data"""
    
    print("\n=== ENHANCED FEATURE ENGINEERING ===")
    
    well_col = df._well_col
    date_col = df._date_col
    numeric_cols = df._numeric_cols
    
    # Sort by well and date
    sort_cols = [well_col]
    if date_col:
        sort_cols.append(date_col)
    
    df = df.sort_values(sort_cols)
    
    well_features = []
    
    for well in df[well_col].unique():
        if pd.isna(well):
            continue
            
        well_data = df[df[well_col] == well].copy()
        
        if len(well_data) == 0:
            continue
            
        # Initialize features dictionary
        features = {'WELL_NAME': well, 'total_records': len(well_data)}
        
        # Time-based features
        if date_col and not well_data[date_col].isna().all():
            features.update(extract_time_features(well_data, date_col))
        
        # Statistical features for each numeric column
        for col in numeric_cols:
            if col in well_data.columns and not well_data[col].isna().all():
                features.update(extract_statistical_features(well_data, col))
        
        # Production trend features
        features.update(extract_trend_features(well_data, numeric_cols))
        
        # Production ratio features
        features.update(extract_ratio_features(well_data, numeric_cols))
        
        # Well performance features
        features.update(extract_performance_features(well_data, numeric_cols))
        
        well_features.append(features)
    
    feature_df = pd.DataFrame(well_features)
    
    # Clean the feature dataframe
    feature_df = feature_df.fillna(0)
    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    
    # Remove constant features
    constant_features = [col for col in feature_df.columns if feature_df[col].nunique() <= 1]
    if constant_features:
        print(f"⚠ Removing {len(constant_features)} constant features")
        feature_df = feature_df.drop(columns=constant_features)
    
    print(f"✓ Engineered features shape: {feature_df.shape}")
    print(f"✓ Total features: {len(feature_df.columns)}")
    
    return feature_df

def extract_time_features(well_data, date_col):
    """Extract time-based features"""
    features = {}
    
    if date_col in well_data.columns:
        dates = well_data[date_col].dropna()
        if len(dates) > 0:
            features['production_days'] = (dates.max() - dates.min()).days
            features['first_production_year'] = dates.min().year
            features['last_production_year'] = dates.max().year
            features['production_span_years'] = (dates.max() - dates.min()).days / 365.25
    
    return features

def extract_statistical_features(well_data, col):
    """Extract statistical features for a column"""
    col_clean = col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
    features = {}
    
    values = well_data[col].dropna()
    if len(values) > 0:
        features[f'avg_{col_clean}'] = values.mean()
        features[f'max_{col_clean}'] = values.max()
        features[f'min_{col_clean}'] = values.min()
        features[f'std_{col_clean}'] = values.std()
        features[f'sum_{col_clean}'] = values.sum()
        features[f'median_{col_clean}'] = values.median()
        features[f'skew_{col_clean}'] = values.skew()
        features[f'kurtosis_{col_clean}'] = values.kurtosis()
        
        # Percentiles
        features[f'q25_{col_clean}'] = values.quantile(0.25)
        features[f'q75_{col_clean}'] = values.quantile(0.75)
        
        # Variability measures
        if values.mean() != 0:
            features[f'cv_{col_clean}'] = values.std() / values.mean()  # Coefficient of variation
    
    return features

def extract_trend_features(well_data, numeric_cols):
    """Extract trend-based features"""
    features = {}
    
    for col in numeric_cols:
        if col in well_data.columns:
            values = well_data[col].dropna()
            if len(values) > 1:
                col_clean = col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
                
                # Linear trend
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                features[f'{col_clean}_slope'] = slope
                features[f'{col_clean}_r_squared'] = r_value**2
                
                # Change metrics
                features[f'{col_clean}_total_change'] = values.iloc[-1] - values.iloc[0]
                features[f'{col_clean}_change_rate'] = (values.iloc[-1] - values.iloc[0]) / len(values)
                
                # Volatility
                if len(values) > 2:
                    pct_change = values.pct_change().dropna()
                    if len(pct_change) > 0:
                        features[f'{col_clean}_volatility'] = pct_change.std()
    
    return features

def extract_ratio_features(well_data, numeric_cols):
    """Extract ratio-based features"""
    features = {}
    
    # Find cumulative production columns
    oil_cols = [col for col in numeric_cols if 'OIL' in col.upper() and 'CUMULATIVE' in col.upper()]
    water_cols = [col for col in numeric_cols if 'WATER' in col.upper() and 'CUMULATIVE' in col.upper()]
    gas_cols = [col for col in numeric_cols if 'GAS' in col.upper() and 'CUMULATIVE' in col.upper()]
    
    # Calculate ratios
    if oil_cols and water_cols:
        oil_prod = well_data[oil_cols[0]].iloc[-1] if len(well_data) > 0 else 0
        water_prod = well_data[water_cols[0]].iloc[-1] if len(well_data) > 0 else 0
        
        if oil_prod > 0:
            features['water_oil_ratio'] = water_prod / oil_prod
        
        total_liquid = oil_prod + water_prod
        if total_liquid > 0:
            features['water_cut'] = water_prod / total_liquid
            features['oil_cut'] = oil_prod / total_liquid
    
    if oil_cols and gas_cols:
        oil_prod = well_data[oil_cols[0]].iloc[-1] if len(well_data) > 0 else 0
        gas_prod = well_data[gas_cols[0]].iloc[-1] if len(well_data) > 0 else 0
        
        if oil_prod > 0:
            features['gas_oil_ratio'] = gas_prod / oil_prod
    
    return features

def extract_performance_features(well_data, numeric_cols):
    """Extract well performance features"""
    features = {}
    
    # Production efficiency metrics
    pressure_cols = [col for col in numeric_cols if 'PRESSURE' in col.upper()]
    production_cols = [col for col in numeric_cols if 'PROD' in col.upper() and 'CUMULATIVE' not in col.upper()]
    
    if pressure_cols and production_cols:
        for prod_col in production_cols:
            for press_col in pressure_cols:
                if prod_col in well_data.columns and press_col in well_data.columns:
                    prod_values = well_data[prod_col].dropna()
                    press_values = well_data[press_col].dropna()
                    
                    if len(prod_values) > 0 and len(press_values) > 0:
                        avg_prod = prod_values.mean()
                        avg_press = press_values.mean()
                        
                        if avg_press > 0:
                            prod_clean = prod_col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
                            press_clean = press_col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
                            features[f'{prod_clean}_per_{press_clean}'] = avg_prod / avg_press
    
    return features

# ============================================================================
# 4. ENHANCED STATISTICAL ANALYSIS
# ============================================================================

def perform_statistical_analysis(features_df, target_df=None):
    """Enhanced statistical analysis with visualization"""
    
    print("\n=== ENHANCED STATISTICAL ANALYSIS ===")
    
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    print(f"✓ Analyzing {len(numeric_features)} numeric features")
    
    # Basic statistics
    stats_summary = features_df[numeric_features].describe()
    print("✓ Basic statistics calculated")
    
    # Correlation analysis
    if len(numeric_features) > 1:
        corr_matrix = features_df[numeric_features].corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print(f"⚠ Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8)")
            for feat1, feat2, corr in high_corr_pairs[:5]:
                print(f"  {feat1} vs {feat2}: {corr:.3f}")
        
        # Feature variance analysis
        low_variance_features = []
        for col in numeric_features:
            if features_df[col].var() < 0.01:
                low_variance_features.append(col)
        
        if low_variance_features:
            print(f"⚠ Found {len(low_variance_features)} low variance features")
        
        return corr_matrix, stats_summary
    
    return None, stats_summary

# ============================================================================
# 5. ENHANCED TARGET VARIABLE CREATION
# ============================================================================

def create_target_variable(classification_params, features_df):
    """Enhanced target variable creation with better handling"""
    
    print("\n=== ENHANCED TARGET VARIABLE CREATION ===")
    
    if classification_params is None:
        print("⚠ No classification parameters provided. Creating synthetic target.")
        return create_synthetic_target(features_df)
    
    # Find well identifier column
    well_id_col = identify_well_column(classification_params)
    if not well_id_col:
        well_id_col = classification_params.columns[0]
    
    print(f"✓ Using '{well_id_col}' as well identifier")
    
    # Find target column
    target_col = identify_target_column(classification_params)
    
    if target_col:
        print(f"✓ Using '{target_col}' as target variable")
        target_df = classification_params[[well_id_col, target_col]].copy()
        target_df['target_class'] = target_df[target_col].astype(str)
    else:
        print("⚠ No clear target column found. Creating composite target.")
        target_df = create_composite_target(classification_params, well_id_col)
    
    # Clean and process target
    target_df = clean_target_data(target_df, well_id_col)
    
    # Encode target
    le = LabelEncoder()
    target_df['target_encoded'] = le.fit_transform(target_df['target_class'])
    
    print(f"✓ Target classes: {target_df['target_class'].unique()}")
    print("✓ Target distribution:")
    print(target_df['target_class'].value_counts())
    
    # Merge with features
    final_df = features_df.merge(target_df[[well_id_col, 'target_class', 'target_encoded']], 
                                left_on='WELL_NAME', right_on=well_id_col, how='inner')
    
    print(f"✓ Final dataset shape: {final_df.shape}")
    print(f"✓ Wells with both features and targets: {len(final_df)}")
    
    return final_df, le

def identify_target_column(df):
    """Identify target column in classification parameters"""
    target_keywords = ['TYPE', 'CLASS', 'CATEGORY', 'LABEL', 'TARGET']
    
    for col in df.columns:
        if any(keyword in col.upper() for keyword in target_keywords):
            return col
    
    return None

def create_synthetic_target(features_df):
    """Create synthetic target based on feature clustering"""
    from sklearn.cluster import KMeans
    
    print("⚠ Creating synthetic target using K-means clustering")
    
    # Use key features for clustering
    key_features = []
    for col in features_df.columns:
        if any(keyword in col.lower() for keyword in ['avg_', 'sum_', 'total_', 'max_']):
            key_features.append(col)
    
    if len(key_features) > 0:
        X = features_df[key_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use elbow method to find optimal k
        optimal_k = min(3, len(features_df) // 2)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        le = LabelEncoder()
        target_encoded = le.fit_transform([f'cluster_{i}' for i in clusters])
        
        final_df = features_df.copy()
        final_df['target_class'] = [f'cluster_{i}' for i in clusters]
        final_df['target_encoded'] = target_encoded
        
        return final_df, le
    
    return None, None

def create_composite_target(classification_params, well_id_col):
    """Create composite target from multiple columns"""
    non_id_cols = [col for col in classification_params.columns if col != well_id_col]
    
    if len(non_id_cols) >= 2:
        target_df = classification_params[[well_id_col]].copy()
        target_df['target_class'] = classification_params[non_id_cols[0]].astype(str) + '_' + classification_params[non_id_cols[1]].astype(str)
    else:
        target_df = classification_params[[well_id_col]].copy()
        target_df['target_class'] = classification_params[non_id_cols[0]].astype(str) if non_id_cols else 'default'
    
    return target_df

def clean_target_data(target_df, well_id_col):
    """Clean target data"""
    # Remove whitespace and handle missing values
    target_df['target_class'] = target_df['target_class'].str.strip()
    target_df = target_df.dropna(subset=['target_class'])
    
    # Remove classes with very few samples
    class_counts = target_df['target_class'].value_counts()
    min_samples = 2
    valid_classes = class_counts[class_counts >= min_samples].index
    target_df = target_df[target_df['target_class'].isin(valid_classes)]
    
    return target_df

# ============================================================================
# 6. ENHANCED MODEL TRAINING AND EVALUATION
# ============================================================================

def train_enhanced_xgboost_model(df, target_col='target_encoded'):
    """Enhanced XGBoost model training with better evaluation"""
    
    print("\n=== ENHANCED MODEL TRAINING ===")
    
    # Prepare features
    exclude_cols = ['WELL_NAME', 'target_class', 'target_encoded']
    for col in df.columns:
        if any(keyword in col for keyword in ['Reservoir', 'Well', 'WELL', 'NAME']):
            exclude_cols.append(col)
    
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_columns].copy()
    y = df[target_col].copy()
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Number of classes: {len(np.unique(y))}")
    
    # Remove constant and near-constant features
    X = remove_low_variance_features(X)
    feature_columns = X.columns.tolist()
    
    # Handle class imbalance
    class_counts = pd.Series(y).value_counts()
    print(f"✓ Class distribution: {class_counts.to_dict()}")
    
    # Split data with stratification
    test_size = min(0.3, 0.8)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features using RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost with class weight balancing
    class_weights = len(y) / (len(np.unique(y)) * np.bincount(y))
    sample_weights = class_weights[y_train]
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist'
    )
    
    # Train model
    xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # Evaluate model
    y_pred = xgb_model.predict(X_test_scaled)
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ F1-Score (weighted): {f1:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, 
                               cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                               scoring='accuracy')
    print(f"✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    print("\n✓ Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n✓ Top 10 Feature Importances:")
    print(feature_importance.head(10))
    
    return xgb_model, scaler, feature_columns, feature_importance

def remove_low_variance_features(X, threshold=0.01):
    """Remove features with low variance"""
    low_variance_cols = []
    for col in X.columns:
        if X[col].var() < threshold:
            low_variance_cols.append(col)
    
    if low_variance_cols:
        print(f"⚠ Removing {len(low_variance_cols)} low variance features")
        X = X.drop(columns=low_variance_cols)
    
    return X

# ============================================================================
# 7. ENHANCED PREDICTION AND OUTPUT
# ============================================================================

def generate_enhanced_predictions(model, scaler, feature_columns, all_wells_df, le):
    """Generate enhanced predictions with confidence metrics"""
    
    print("\n=== GENERATING ENHANCED PREDICTIONS ===")
    
    # Prepare features
    X_all = all_wells_df[feature_columns]
    X_all_scaled = scaler.transform(X_all)
    
    # Make predictions
    predictions = model.predict(X_all_scaled)
    prediction_proba = model.predict_proba(X_all_scaled)
    
    # Calculate prediction confidence
    max_proba = np.max(prediction_proba, axis=1)
    prediction_confidence = np.where(max_proba > 0.5, 'High', 
                                   np.where(max_proba > 0.3, 'Medium', 'Low'))
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'WELL_NAME': all_wells_df['WELL_NAME'],
        '