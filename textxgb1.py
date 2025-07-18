"""
Data Phandas SPE DSEATS Africa Datathon 2025 - Well Classification Pipeline with XGBoost
Complete pipeline from data cleaning to XGBoost classification with hyperparameter tuning
Now includes Formation Volume Factor (FVF) and reservoir barrel calculations
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WellClassificationPipeline:
    """
    Complete pipeline for well classification using XGBoost with hyperparameter tuning
    """
    
    def __init__(self):
        self.wells_data = None
        self.reservoir_info = None
        self.classification_params = None
        self.processed_features = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.best_params = {}
        self.target_columns = [
            'Reservoir Name', 'Reservoir Type', 'Well Type', 'Production Type',
            'Formation GOR Trend', 'Watercut Trend', 'Oil Productivity Index Trend'
        ]
        
        # Initialize reservoir FVF mapping (will be populated in load_data)
        self.reservoir_fvf = None
        
    def load_data(self):
        """Load and examine the datasets"""
        print("Loading datasets...")
        
        # Load wells production data
        self.wells_data = pd.read_csv('data/spe_africa_dseats_datathon_2025_wells_dataset.csv')
        
        # Load reservoir information
        self.reservoir_info = pd.read_csv('data/reservoir_info.csv')
        
        # Create FVF mapping (RB/STB) from reservoir info
        self.reservoir_fvf = dict(zip(
            self.reservoir_info['Reservoir Name'],
            self.reservoir_info['Formation Volume Factor (RB/STB)']
        ))
        
        # Load classification parameters (target classes)
        self.classification_params = pd.read_csv('data/classification_parameters.csv')
        
        print(f"Wells data shape: {self.wells_data.shape}")
        print(f"Reservoir info shape: {self.reservoir_info.shape}")
        print(f"Classification params shape: {self.classification_params.shape}")
        
        # Display basic info
        print("\nWells data columns:")
        print(self.wells_data.columns.tolist())
        print(f"\nUnique wells: {self.wells_data['WELL_NAME'].nunique()}")
        print(f"Date range: {self.wells_data['PROD_DATE'].min()} to {self.wells_data['PROD_DATE'].max()}")
        print("\nReservoir FVF values:")
        print(self.reservoir_fvf)
        
    def clean_data(self):
        """Clean and preprocess the raw data"""
        print("\nCleaning data...")
        
        # Convert date column
        self.wells_data['PROD_DATE'] = pd.to_datetime(self.wells_data['PROD_DATE'])
        
        # Clean numeric columns that might have commas
        numeric_columns = [
            'BOTTOMHOLE_FLOWING_PRESSURE (PSI)',
            'ANNULUS_PRESS (PSI)', 
            'WELL_HEAD_PRESSURE (PSI)',
            'CUMULATIVE_OIL_PROD (STB)',
            'CUMULATIVE_FORMATION_GAS_PROD (MSCF)',
            'CUMULATIVE_TOTAL_GAS_PROD (MSCF)',
            'CUMULATIVE_WATER_PROD (BBL)'
        ]
        
        for col in numeric_columns:
            if col in self.wells_data.columns:
                # Remove commas and convert to numeric
                self.wells_data[col] = pd.to_numeric(
                    self.wells_data[col].astype(str).str.replace(',', ''),
                    errors='coerce'
                )
        
        # Handle missing values
        self.wells_data = self.wells_data.fillna(0)
        
        # Sort by well name and date
        self.wells_data = self.wells_data.sort_values(['WELL_NAME', 'PROD_DATE'])
        
        print("Data cleaning completed!")
        print(f"Final dataset shape: {self.wells_data.shape}")
        
    def feature_engineering(self):
        """Engineer features from time series production data"""
        print("\nEngineering features...")
        
        features_list = []
        
        for well in self.wells_data['WELL_NAME'].unique():
            well_data = self.wells_data[self.wells_data['WELL_NAME'] == well].copy()
            well_data = well_data.sort_values('PROD_DATE')
            
            # Calculate daily production rates
            well_data['Daily_Oil_Prod'] = well_data['CUMULATIVE_OIL_PROD (STB)'].diff().fillna(0)
            well_data['Daily_Gas_Prod'] = well_data['CUMULATIVE_FORMATION_GAS_PROD (MSCF)'].diff().fillna(0)
            well_data['Daily_Water_Prod'] = well_data['CUMULATIVE_WATER_PROD (BBL)'].diff().fillna(0)
            
            # Calculate GOR (Gas Oil Ratio)
            well_data['GOR'] = np.where(
                well_data['Daily_Oil_Prod'] > 0,
                well_data['Daily_Gas_Prod'] / well_data['Daily_Oil_Prod'] * 1000,  # Convert to SCF/STB
                0
            )
            
            # Calculate water cut
            total_liquid = well_data['Daily_Oil_Prod'] + well_data['Daily_Water_Prod']
            well_data['Water_Cut'] = np.where(
                total_liquid > 0,
                well_data['Daily_Water_Prod'] / total_liquid * 100,
                0
            )
            
            # Calculate productivity index (approximate)
            avg_reservoir_pressure = 3500  # We'll refine this later
            well_data['Productivity_Index'] = np.where(
                well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)'] > 0,
                well_data['Daily_Oil_Prod'] / (avg_reservoir_pressure - well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)']),
                0
            )
            
            # Identify reservoir for FVF lookup
            reservoir_name = self._identify_reservoir({
                'Avg_BHP': well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)'].mean()
            })
            
            # Calculate reservoir barrels
            cumulative_oil_stb = well_data['CUMULATIVE_OIL_PROD (STB)'].max()
            cumulative_oil_rb = cumulative_oil_stb * self.reservoir_fvf[reservoir_name]
            
            # Aggregate features for the well
            well_features = {
                'WELL_NAME': well,
                'Well_Number': int(well.split('_#')[1]),
                
                # Production statistics (both STB and RB)
                'Total_Oil_STB': cumulative_oil_stb,
                'Total_Oil_RB': cumulative_oil_rb,
                'Reservoir_FVF': self.reservoir_fvf[reservoir_name],
                'Total_Gas_Prod': well_data['CUMULATIVE_FORMATION_GAS_PROD (MSCF)'].max(),
                'Total_Water_Prod': well_data['CUMULATIVE_WATER_PROD (BBL)'].max(),
                'Avg_Daily_Oil': well_data['Daily_Oil_Prod'].mean(),
                'Max_Daily_Oil': well_data['Daily_Oil_Prod'].max(),
                'Oil_Production_Decline': self._calculate_decline_rate(well_data['Daily_Oil_Prod']),
                
                # Pressure statistics
                'Avg_BHP': well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)'].mean(),
                'Max_BHP': well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)'].max(),
                'Min_BHP': well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)'].min(),
                'BHP_Variance': well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)'].var(),
                'Avg_WHP': well_data['WELL_HEAD_PRESSURE (PSI)'].mean(),
                'Avg_Annulus_Press': well_data['ANNULUS_PRESS (PSI)'].mean(),
                
                # Temperature statistics
                'Avg_Downhole_Temp': well_data['DOWNHOLE_TEMPERATURE (deg F)'].mean(),
                'Avg_Wellhead_Temp': well_data['WELL_HEAD_TEMPERATURE (deg F)'].mean(),
                
                # Flow characteristics
                'Avg_Choke_Size': well_data['CHOKE_SIZE (%)'].mean(),
                'Choke_Variance': well_data['CHOKE_SIZE (%)'].var(),
                'Avg_Onstream_Hours': well_data['ON_STREAM_HRS'].mean(),
                
                # GOR and Water Cut trends
                'Avg_GOR': well_data['GOR'].mean(),
                'GOR_Trend': self._calculate_trend(well_data['GOR']),
                'Max_GOR': well_data['GOR'].max(),
                'Avg_Water_Cut': well_data['Water_Cut'].mean(),
                'Water_Cut_Trend': self._calculate_trend(well_data['Water_Cut']),
                'Max_Water_Cut': well_data['Water_Cut'].max(),
                
                # Productivity Index
                'Avg_PI': well_data['Productivity_Index'].mean(),
                'PI_Trend': self._calculate_trend(well_data['Productivity_Index']),
                'Max_PI': well_data['Productivity_Index'].max(),
                
                # Production stability
                'Production_Stability': self._calculate_stability(well_data['Daily_Oil_Prod']),
                'Days_Produced': len(well_data[well_data['Daily_Oil_Prod'] > 0]),
                'Production_Efficiency': well_data['ON_STREAM_HRS'].mean() / 24.0,
                
                # Well type indicators
                'Has_Annulus_Pressure': (well_data['ANNULUS_PRESS (PSI)'] > 0).any(),
                'Pressure_Differential': well_data['WELL_HEAD_PRESSURE (PSI)'].mean() - well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)'].mean(),
                
                # Reservoir info
                'Reservoir_Name': reservoir_name
            }
            
            features_list.append(well_features)
        
        self.processed_features = pd.DataFrame(features_list)
        print(f"Feature engineering completed! Shape: {self.processed_features.shape}")
        print(f"Features created: {len(self.processed_features.columns)}")
        
    def calculate_reservoir_production(self):
        """Calculate total oil produced per reservoir in reservoir barrels (RB)"""
        if not hasattr(self, 'processed_features'):
            raise ValueError("Run feature_engineering() first")
            
        # Group by reservoir and sum production
        reservoir_production = self.processed_features.groupby('Reservoir_Name').agg({
            'Total_Oil_STB': 'sum',
            'Total_Oil_RB': 'sum'
        }).rename(columns={
            'Total_Oil_STB': 'Total_Surface_Barrels',
            'Total_Oil_RB': 'Total_Reservoir_Barrels'
        })
        
        return reservoir_production

    def _calculate_decline_rate(self, production_series):
        """Calculate production decline rate"""
        if len(production_series) < 2:
            return 0
        
        # Use linear regression to estimate decline
        x = np.arange(len(production_series))
        y = production_series.values
        
        # Remove zeros for log calculation
        y_nonzero = y[y > 0]
        if len(y_nonzero) < 2:
            return 0
        
        # Simple linear decline
        if len(y_nonzero) == len(y):
            slope = np.polyfit(x, y, 1)[0]
            return slope / np.mean(y) if np.mean(y) > 0 else 0
        else:
            return 0
    
    def _calculate_trend(self, series):
        """Calculate trend direction: 1 for increasing, -1 for decreasing, 0 for flat"""
        if len(series) < 2:
            return 0
        
        # Remove outliers using IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        filtered_series = series[(series >= Q1 - 1.5*IQR) & (series <= Q3 + 1.5*IQR)]
        
        if len(filtered_series) < 2:
            return 0
        
        # Calculate trend using linear regression
        x = np.arange(len(filtered_series))
        slope = np.polyfit(x, filtered_series.values, 1)[0]
        
        # Threshold for significance
        threshold = 0.01 * np.mean(filtered_series)
        
        if slope > threshold:
            return 1  # Increasing
        elif slope < -threshold:
            return -1  # Decreasing
        else:
            return 0  # Flat
    
    def _calculate_stability(self, production_series):
        """Calculate production stability (coefficient of variation)"""
        if len(production_series) < 2 or production_series.mean() == 0:
            return 0
        
        # Calculate coefficient of variation
        cv = production_series.std() / production_series.mean()
        return 1 / (1 + cv)  # Normalize to 0-1 where 1 is most stable
    
    def prepare_target_variables(self):
        """Prepare target variables for classification"""
        print("\nPreparing target variables...")
        
        # Create a mapping from well number to classifications
        target_df = pd.DataFrame({
            'Well_Number': range(1, 21),
            'Reservoir Name': None,
            'Reservoir Type': None,
            'Well Type': None,
            'Production Type': None,
            'Formation GOR Trend': None,
            'Watercut Trend': None,
            'Oil Productivity Index Trend': None
        })
        
        # Implement classification logic based on reservoir engineering rules
        for idx, row in self.processed_features.iterrows():
            well_num = row['Well_Number']
            
            # Reservoir identification based on BHP and reservoir pressures
            reservoir_name = self._identify_reservoir(row)
            target_df.loc[well_num-1, 'Reservoir Name'] = reservoir_name
            
            # Reservoir type based on reservoir info
            reservoir_type = self._determine_reservoir_type(reservoir_name)
            target_df.loc[well_num-1, 'Reservoir Type'] = reservoir_type
            
            # Well type based on annulus pressure
            well_type = 'GL' if row['Has_Annulus_Pressure'] else 'NF'
            target_df.loc[well_num-1, 'Well Type'] = well_type
            
            # Production type based on stability
            prod_type = 'Steady' if row['Production_Stability'] > 0.5 else 'Unsteady'
            target_df.loc[well_num-1, 'Production Type'] = prod_type
            
            # GOR trend classification
            gor_trend = self._classify_gor_trend(row, reservoir_name)
            target_df.loc[well_num-1, 'Formation GOR Trend'] = gor_trend
            
            # Water cut trend
            wc_trend = self._classify_watercut_trend(row['Water_Cut_Trend'])
            target_df.loc[well_num-1, 'Watercut Trend'] = wc_trend
            
            # PI trend
            pi_trend = self._classify_pi_trend(row['PI_Trend'])
            target_df.loc[well_num-1, 'Oil Productivity Index Trend'] = pi_trend
        
        self.target_data = target_df
        print("Target variables prepared!")
        
    def _identify_reservoir(self, well_features):
        """Identify reservoir based on pressure characteristics"""
        bhp = well_features['Avg_BHP']
        
        # Reservoir pressure ranges (with 200 psi differential allowance)
        reservoir_pressures = {
            'ACHI': 2700,
            'KEMA': 3900,
            'MAKO': 3000,
            'DEPU': 2400,
            'JANI': 4200
        }
        
        # Find closest match within 200 psi
        min_diff = float('inf')
        best_reservoir = 'ACHI'
        
        for reservoir, pressure in reservoir_pressures.items():
            diff = abs(bhp - pressure)
            if diff < min_diff and diff <= 200:
                min_diff = diff
                best_reservoir = reservoir
        
        return best_reservoir
    
    def _determine_reservoir_type(self, reservoir_name):
        """Determine if reservoir is saturated or undersaturated"""
        reservoir_types = {
            'ACHI': 'Saturated',
            'KEMA': 'Undersat',
            'MAKO': 'Saturated',
            'DEPU': 'Saturated',
            'JANI': 'Undersat'
        }
        return reservoir_types.get(reservoir_name, 'Saturated')
    
    def _classify_gor_trend(self, well_features, reservoir_name):
        """Classify GOR trend relative to solution GOR"""
        solution_gor = {
            'ACHI': 800,
            'KEMA': 600,
            'MAKO': 500,
            'DEPU': 1200,
            'JANI': 1000
        }
        
        avg_gor = well_features['Avg_GOR']
        sol_gor = solution_gor.get(reservoir_name, 800)
        
        if avg_gor > sol_gor * 1.1:
            return 'aSolGOR'
        elif avg_gor < sol_gor * 0.9:
            return 'bSolGOR'
        else:
            return 'Combo'
    
    def _classify_watercut_trend(self, trend_value):
        """Classify water cut trend"""
        if trend_value > 0.5:
            return 'Incr'
        elif trend_value < -0.5:
            return 'Decr'
        else:
            return 'Flat'
    
    def _classify_pi_trend(self, trend_value):
        """Classify productivity index trend"""
        if trend_value > 0.5:
            return 'Incr'
        elif trend_value < -0.5:
            return 'Decr'
        else:
            return 'Flat'
    
    def prepare_features_for_ml(self):
        """Prepare features for machine learning"""
        print("\nPreparing features for ML...")
        
        # Select numerical features for ML - CORRECTED COLUMN NAMES
        feature_columns = [
            'Total_Oil_STB',  # Changed from 'Total_Oil_Prod'
            'Total_Gas_Prod', 'Total_Water_Prod',
            'Avg_Daily_Oil', 'Max_Daily_Oil', 'Oil_Production_Decline',
            'Avg_BHP', 'Max_BHP', 'Min_BHP', 'BHP_Variance',
            'Avg_WHP', 'Avg_Annulus_Press',
            'Avg_Downhole_Temp', 'Avg_Wellhead_Temp',
            'Avg_Choke_Size', 'Choke_Variance', 'Avg_Onstream_Hours',
            'Avg_GOR', 'GOR_Trend', 'Max_GOR',
            'Avg_Water_Cut', 'Water_Cut_Trend', 'Max_Water_Cut',
            'Avg_PI', 'PI_Trend', 'Max_PI',
            'Production_Stability', 'Days_Produced', 'Production_Efficiency',
            'Pressure_Differential'
        ]
        
        # Convert boolean to numeric
        self.processed_features['Has_Annulus_Pressure'] = self.processed_features['Has_Annulus_Pressure'].astype(int)
        feature_columns.append('Has_Annulus_Pressure')
        
        # Debug: Print available columns vs requested columns
        print("Available columns in processed_features:")
        print(self.processed_features.columns.tolist())
        print("\nRequested feature columns:")
        print(feature_columns)
        
        # Check which columns are missing
        missing_columns = [col for col in feature_columns if col not in self.processed_features.columns]
        if missing_columns:
            print(f"\nMissing columns: {missing_columns}")
            # Remove missing columns from feature_columns
            feature_columns = [col for col in feature_columns if col in self.processed_features.columns]
            print(f"Using available columns: {feature_columns}")
        
        # Create feature matrix
        X = self.processed_features[feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        print(f"Features prepared! Shape: {self.X_scaled.shape}")
        return feature_columns  # Return the actual columns used

    def generate_classification_results(self):
        """Generate final classification results and production summary"""
        print("\nGenerating classification results...")
        
        # Make predictions
        results = []
        for well_num in range(1, 21):
            well_result = {'Well': well_num}
            features = self.X_scaled[self.X_scaled.index == well_num - 1]
            
            for target_col, model in self.models.items():
                le = self.label_encoders[target_col]
                pred = model.predict(features)[0]
                well_result[target_col] = le.inverse_transform([pred])[0]
            
            results.append(well_result)
        
        results_df = pd.DataFrame(results)
        
        # Calculate reservoir production totals
        reservoir_prod = self.calculate_reservoir_production()
        
        # Save results
        results_df.to_csv('Data_Phandas_DSEATS_Africa_2025_Classification.csv', index=False)
        reservoir_prod.to_csv('Data_Phandas_DSEATS_Africa_2025_ReservoirProduction.csv', index=False)
        
        print("Classification results generated and saved!")
        print("\nReservoir Production Summary:")
        print(reservoir_prod)
        
        return results_df, reservoir_prod

    def train_xgboost_model(self):
        """Train XGBoost model with hyperparameter tuning"""
        print("\nTraining XGBoost model with hyperparameter tuning...")

        # Prepare target variables
        y_encoded = {}
        valid_targets = []

        # First pass to check which targets have multiple classes
        for target_col in self.target_columns:
            le = LabelEncoder()
            y_encoded[target_col] = le.fit_transform(self.target_data[target_col].astype(str))
            self.label_encoders[target_col] = le
            
            if len(np.unique(y_encoded[target_col])) > 1:
                valid_targets.append(target_col)
            else:
                print(f"Skipping {target_col} - only one class present")

        if not valid_targets:
            raise ValueError("No valid targets with multiple classes for classification")

        # Parameter space for tuning
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }

        # Train individual models for each target
        self.models = {}
        
        for target_col in valid_targets:
            print(f"\nTuning XGBoost for target: {target_col}")
            
            n_classes = len(np.unique(y_encoded[target_col]))
            
            model = XGBClassifier(
                objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
                num_class=n_classes if n_classes > 2 else None,
                eval_metric='mlogloss' if n_classes > 2 else 'logloss',
                use_label_encoder=False,
                verbosity=0,
                random_state=42
            )

            # Use stratified k-fold for classification
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Compute sample weights for class imbalance
            sample_weights = compute_sample_weight('balanced', y_encoded[target_col])
            
            # Randomized search for hyperparameter tuning
            search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=20,
                cv=cv,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            search.fit(self.X_scaled, y_encoded[target_col], sample_weight=sample_weights)
            self.models[target_col] = search.best_estimator_
            self.best_params[target_col] = search.best_params_

            print(f"Best params for {target_col}: {search.best_params_}")
            print(f"Best CV score: {search.best_score_:.3f}")

        # Evaluate performance on training data
        accuracies = {}
        for target_col in valid_targets:
            y_pred = self.models[target_col].predict(self.X_scaled)
            acc = accuracy_score(y_encoded[target_col], y_pred)
            accuracies[target_col] = acc
            print(f"Accuracy for {target_col}: {acc:.3f}")

        print(f"\nOverall mean accuracy: {np.mean(list(accuracies.values())):.3f}")
        
    def explain_xgboost_model(self):
        """Explain XGBoost model and feature importance"""
        print("\n" + "="*50)
        print("XGBOOST MODEL EXPLANATION")
        print("="*50)
        
        print("""
        XGBOOST OVERVIEW:
        ================
        
        XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting framework
        that uses an ensemble of decision trees. It builds models sequentially, where
        each new model corrects errors made by previous models.
        
        MATHEMATICAL FOUNDATION:
        ======================
        
        XGBoost minimizes the objective function:
        Obj = Σ L(yi, ŷi) + Σ Ω(fk)
        
        Where:
        - L(yi, ŷi) is the loss function (e.g., logistic loss for classification)
        - Ω(fk) is the regularization term to prevent overfitting
        - fk represents individual trees
        
        KEY FEATURES:
        ============
        1. Gradient Boosting: Sequential model improvement
        2. Regularization: L1 and L2 regularization to prevent overfitting
        3. Tree Pruning: Removes unnecessary branches
        4. Parallel Processing: Efficient computation
        5. Missing Value Handling: Automatic handling of missing values
        
        HYPERPARAMETERS TUNED:
        =====================
        """)
        
        # Display best parameters for each target
        for target_col, params in self.best_params.items():
            print(f"\n{target_col}:")
            for param, value in params.items():
                print(f"  - {param}: {value}")
        
        print(f"""
        
        ADVANTAGES FOR WELL CLASSIFICATION:
        =================================
        1. Handles non-linear relationships between features
        2. Automatic feature selection through tree splits
        3. Robust to outliers and missing values
        4. Provides feature importance scores
        5. Excellent performance on tabular data
        6. Handles mixed data types (numerical and categorical)
        
        FEATURE IMPORTANCE:
        ==================
        """)
        
        # Calculate and display feature importance
        self._display_feature_importance()
        
    def _display_feature_importance(self):
        """Display feature importance for each target"""
        feature_names = self.X_scaled.columns
        
        for target_col, model in self.models.items():
            print(f"\nTop 10 Important Features for {target_col}:")
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Display top 10 features
                for j, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    print(f"  {j+1}. {row['feature']}: {row['importance']:.4f}")
    
    def generate_classification_results(self):
        """Generate final classification results"""
        print("\nGenerating classification results...")
        
        # Initialize results dataframe
        results = []
        
        for well_num in self.processed_features['Well_Number']:
            well_result = {'Well': well_num}
            features = self.X_scaled[self.X_scaled.index == well_num - 1]
            
            for target_col, model in self.models.items():
                le = self.label_encoders[target_col]
                
                # Predict and decode
                pred = model.predict(features)[0]
                well_result[target_col] = le.inverse_transform([pred])[0]
                
            results.append(well_result)
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv('Data_Phandas1_DSEATS_Africa_2025_XGBoost_Classification.csv', index=False)
        
        print("Classification results generated and saved!")
        print(results_df.head())
        
        return results_df
    
    def visualize_results(self):
        """Create visualizations including reservoir production"""
        # Set up the plotting
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Production Profile by Well
        ax1 = axes[0, 0]
        for well in self.processed_features['Well_Number'].head(5):
            well_data = self.wells_data[self.wells_data['WELL_NAME'] == f'WELL_#{well}']
            ax1.plot(well_data['PROD_DATE'], well_data['CUMULATIVE_OIL_PROD (STB)'], 
                    label=f'Well #{well}', linewidth=2)
        ax1.set_title('Cumulative Oil Production by Well', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Oil (STB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pressure Distribution
        ax2 = axes[0, 1]
        ax2.boxplot([self.processed_features['Avg_BHP'], 
                    self.processed_features['Avg_WHP']], 
                   labels=['BHP', 'WHP'])
        ax2.set_title('Pressure Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Pressure (PSI)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: GOR vs Water Cut
        ax3 = axes[0, 2]
        scatter = ax3.scatter(self.processed_features['Avg_GOR'], 
                             self.processed_features['Avg_Water_Cut'],
                             c=self.processed_features['Well_Number'], 
                             cmap='viridis', s=100, alpha=0.7)
        ax3.set_title('GOR vs Water Cut', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Average GOR (SCF/STB)')
        ax3.set_ylabel('Average Water Cut (%)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Well Number')
        
        # Plot 4: Production Stability
        ax4 = axes[1, 0]
        ax4.hist(self.processed_features['Production_Stability'], 
                bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_title('Production Stability Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Production Stability Index')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Reservoir Classification
        ax5 = axes[1, 1]
        reservoir_counts = self.target_data['Reservoir Name'].value_counts()
        ax5.pie(reservoir_counts.values, labels=reservoir_counts.index, 
               autopct='%1.1f%%', startangle=90)
        ax5.set_title('Reservoir Distribution', fontsize=14, fontweight='bold')
        
        # Plot 6: Feature Correlation Heatmap
        ax6 = axes[1, 2]
        important_features = ['Avg_BHP', 'Avg_GOR', 'Avg_Water_Cut', 'Production_Stability', 
                            'Avg_Daily_Oil', 'Pressure_Differential']
        
        corr_matrix = self.processed_features[important_features].corr()
        im = ax6.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax6.set_xticks(range(len(important_features)))
        ax6.set_yticks(range(len(important_features)))
        ax6.set_xticklabels([feat.replace('_', ' ') for feat in important_features], 
                          rotation=45, ha='right')
        ax6.set_yticklabels([feat.replace('_', ' ') for feat in important_features])
        ax6.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Add correlation values to heatmap
        for i in range(len(important_features)):
            for j in range(len(important_features)):
                text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig('well_classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create additional detailed visualizations
        self._create_detailed_visualizations()
        
        print("Visualizations created and saved!")
    
    def _create_detailed_visualizations(self):
        """Create additional detailed visualizations"""
        
        # Classification Results Summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Well Type Distribution
        ax1 = axes[0, 0]
        well_type_counts = self.target_data['Well Type'].value_counts()
        colors = ['#FF9999', '#66B2FF']
        ax1.pie(well_type_counts.values, labels=well_type_counts.index, 
               autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title('Well Type Distribution', fontsize=14, fontweight='bold')
        
        # Production Type Distribution
        ax2 = axes[0, 1]
        prod_type_counts = self.target_data['Production Type'].value_counts()
        colors = ['#99FF99', '#FFB366']
        ax2.pie(prod_type_counts.values, labels=prod_type_counts.index, 
               autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Production Type Distribution', fontsize=14, fontweight='bold')
        
        # GOR Trend Classification
        ax3 = axes[1, 0]
        gor_trend_counts = self.target_data['Formation GOR Trend'].value_counts()
        ax3.bar(gor_trend_counts.index, gor_trend_counts.values, 
               color=['#FF99CC', '#99CCFF', '#FFFF99'])
        ax3.set_title('Formation GOR Trend Classification', fontsize=14, fontweight='bold')
        ax3.set_xlabel('GOR Trend')
        ax3.set_ylabel('Number of Wells')
        ax3.grid(True, alpha=0.3)
        
        # Water Cut Trend
        ax4 = axes[1, 1]
        wc_trend_counts = self.target_data['Watercut Trend'].value_counts()
        ax4.bar(wc_trend_counts.index, wc_trend_counts.values, 
               color=['#FFB3BA', '#BAFFC9', '#BAE1FF'])
        ax4.set_title('Water Cut Trend Classification', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Water Cut Trend')
        ax4.set_ylabel('Number of Wells')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('classification_results_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Model Performance Visualization
        self._plot_model_performance()
    
    def _plot_model_performance(self):
        """Plot model performance metrics"""
        
        # Create performance summary
        accuracies = []
        for target_col, model in self.models.items():
            y_true = self.label_encoders[target_col].transform(self.target_data[target_col])
            y_pred = model.predict(self.X_scaled)
            acc = accuracy_score(y_true, y_pred)
            accuracies.append(acc)
        
        # Plot accuracy by target
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))
        bars = ax1.bar(range(len(self.models)), accuracies, color=colors)
        ax1.set_xlabel('Classification Target')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('XGBoost Model Accuracy by Target', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(self.models)))
        ax1.set_xticklabels([col.replace(' ', '\n') for col in self.models.keys()], 
                          rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Feature importance summary (average across all targets)
        feature_names = self.X_scaled.columns
        avg_importance = np.zeros(len(feature_names))
        
        for model in self.models.values():
            if hasattr(model, 'feature_importances_'):
                avg_importance += model.feature_importances_
        
        avg_importance /= len(self.models)
        
        # Plot top 10 features
        top_indices = np.argsort(avg_importance)[-10:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = avg_importance[top_indices]
        
        ax2.barh(range(len(top_features)), top_importance, color='skyblue')
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels([feat.replace('_', ' ') for feat in top_features])
        ax2.set_xlabel('Average Feature Importance')
        ax2.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()    
        
        # Add reservoir production plot
        reservoir_prod = self.calculate_reservoir_production()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        reservoir_prod['Total_Reservoir_Barrels'].plot(
            kind='bar', 
            color='darkorange',
            ax=ax,
            title='Total Oil Production by Reservoir (Reservoir Barrels)'
        )
        ax.set_ylabel('Barrels (RB)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reservoir_production_summary.png', dpi=300)
        plt.show()

    def save_model_and_results(self):
        """Save the trained model and all results"""
        print("\nSaving model and results...")
    
    # Save the trained model
        model_data = {
            'model': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'best_params': self.best_params,
            'feature_columns': self.X_scaled.columns.tolist(),
            'target_columns': self.target_columns
        }
        
        with open('xgboost_well_classification_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save processed features
        self.processed_features.to_csv('processed_well_features.csv', index=False)
        
        # Save target data
        self.target_data.to_csv('target_classifications.csv', index=False)
        
        # Create model summary report
        self._create_model_report()
        
        print("Model and results saved successfully!")
    
    def _create_model_report(self):
        """Create a comprehensive model report"""
        
        report = f"""
    ================================================================================
                            XGBOOST WELL CLASSIFICATION MODEL REPORT
    ================================================================================

    MODEL OVERVIEW:
    ==============
    - Algorithm: XGBoost (eXtreme Gradient Boosting)
    - Problem Type: Multi-output Classification
    - Number of Wells: {len(self.processed_features)}
    - Number of Features: {len(self.X_scaled.columns)}
    - Number of Targets: {len(self.target_columns)}
    - Targets Modeled: {len(self.models)}/{len(self.target_columns)}

    CLASSIFICATION TARGETS:
    ======================
    """
        
        for i, target in enumerate(self.target_columns, 1):
            unique_classes = self.target_data[target].unique()
            modeled = "(Modeled)" if target in self.models else "(Skipped - single class)"
            report += f"{i}. {target}: {len(unique_classes)} classes {list(unique_classes)} {modeled}\n"
        
        report += f"""

    HYPERPARAMETER TUNING RESULTS:
    ==============================
    """
        
        for target, params in self.best_params.items():
            report += f"\n{target}:\n"
            for param, value in params.items():
                report += f"  - {param}: {value}\n"
        
        report += f"""

    MODEL PERFORMANCE:
    ==================
    """
        
        # Calculate and add performance metrics
        accuracies = {}
        for target_col, model in self.models.items():
            y_true = self.label_encoders[target_col].transform(self.target_data[target_col])
            y_pred = model.predict(self.X_scaled)
            y_pred_labels = self.label_encoders[target_col].inverse_transform(y_pred)
            acc = accuracy_score(y_true, y_pred)
            report += f"- {target_col}: {acc:.3f}\n"
            accuracies[target_col] = acc
        
        if accuracies:
            overall_acc = np.mean(list(accuracies.values()))
            report += f"\nOverall Average Accuracy: {overall_acc:.3f}\n"
        else:
            report += "\nNo models were trained (all targets had only one class)\n"
        
        report += f"""

    FEATURE ENGINEERING:
    ===================
    - Production statistics (cumulative and daily rates)
    - Pressure characteristics (BHP, WHP, annulus pressure)
    - Flow parameters (GOR, water cut, productivity index)
    - Temperature measurements
    - Production stability and efficiency metrics
    - Trend analysis for key parameters

    MODEL ADVANTAGES:
    =================
    1. Handles non-linear relationships between reservoir parameters
    2. Automatic feature selection through tree-based splits
    3. Robust to outliers and missing values in production data
    4. Provides interpretable feature importance rankings
    5. Excellent performance on tabular reservoir engineering data
    6. Hyperparameter tuning for optimal performance

    RECOMMENDATIONS:
    ===============
    1. Regular model retraining with new production data
    2. Feature engineering refinement based on domain expertise
    3. Cross-validation with additional well data for validation
    4. Integration with reservoir simulation models
    5. Continuous monitoring of model performance in production

    FILES GENERATED:
    ===============
    - Team_DSEATS_Africa_2025_XGBoost_Classification.csv: Final classifications
    - xgboost_well_classification_model.pkl: Trained model
    - processed_well_features.csv: Engineered features
    - target_classifications.csv: Target variable mappings
    - well_classification_analysis.png: EDA visualizations
    - classification_results_summary.png: Results summary
    - model_performance_summary.png: Performance metrics

    ================================================================================
                                        END OF REPORT
    ================================================================================
    """
        
        # Save report
        with open('XGBoost_Well_Classification_Report.txt', 'w') as f:
            f.write(report)
        
        print("Model report generated and saved!")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("="*60)
        print("SPE DSEATS AFRICA 2025 - WELL CLASSIFICATION PIPELINE")
        print("="*60)
        
        # Execute pipeline steps
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        self.prepare_target_variables()
        self.prepare_features_for_ml()
        self.train_xgboost_model()
        self.explain_xgboost_model()
        
        # Generate results
        results = self.generate_classification_results()
        
        # Create visualizations
        self.visualize_results()
        
        # Save everything
        self.save_model_and_results()
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return results

# Main execution
if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = WellClassificationPipeline()
    results = pipeline.run_complete_pipeline()

    reservoir_totals = pipeline.calculate_reservoir_production()
    print(reservoir_totals)
    
    print("\nFinal Classification Results:")
    print(results.to_string(index=False))

        