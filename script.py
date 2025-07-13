"""
SPE DSEATS Africa Datathon 2025 - Well Classification Pipeline
Complete pipeline from data cleaning to Gaussian Naive Bayes classification
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WellClassificationPipeline:
    """
    Complete pipeline for well classification using Gaussian Naive Bayes
    """
    
    def __init__(self):
        self.wells_data = None
        self.reservoir_info = None
        self.classification_params = None
        self.processed_features = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.target_columns = [
            'Reservoir Name', 'Reservoir Type', 'Well Type', 'Production Type',
            'Formation GOR Trend', 'Watercut Trend', 'Oil Productivity Index Trend'
        ]
        
    def load_data(self):
        """Load and examine the datasets"""
        print("Loading datasets...")
        
        # Load wells production data
        self.wells_data = pd.read_csv('data/spe_africa_dseats_datathon_2025_wells_dataset.csv')
        
        # Load reservoir information
        self.reservoir_info = pd.read_csv('data/reservoir_info.csv')
        
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
            # PI = Oil Rate / (Static Pressure - BHP)
            # Assuming average reservoir pressure as static pressure
            avg_reservoir_pressure = 3500  # We'll refine this later
            well_data['Productivity_Index'] = np.where(
                well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)'] > 0,
                well_data['Daily_Oil_Prod'] / (avg_reservoir_pressure - well_data['BOTTOMHOLE_FLOWING_PRESSURE (PSI)']),
                0
            )
            
            # Aggregate features for the well
            well_features = {
                'WELL_NAME': well,
                'Well_Number': int(well.split('_#')[1]),
                
                # Production statistics
                'Total_Oil_Prod': well_data['CUMULATIVE_OIL_PROD (STB)'].max(),
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
            }
            
            features_list.append(well_features)
        
        self.processed_features = pd.DataFrame(features_list)
        print(f"Feature engineering completed! Shape: {self.processed_features.shape}")
        print(f"Features created: {len(self.processed_features.columns)}")
        
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
        # This would normally come from domain knowledge or labeled data
        # For now, we'll use the classification parameters as a reference
        
        # Initialize target dataframe
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
        
        # This is where you would implement the classification logic
        # based on the reservoir engineering rules provided in the challenge
        
        # For demonstration, let's implement some basic rules:
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
        # Based on reservoir info provided
        reservoir_types = {
            'ACHI': 'Saturated',
            'KEMA': 'Undersat',
            'MAKO': 'Saturated',  # Assuming based on bubble point = initial pressure
            'DEPU': 'Saturated',
            'JANI': 'Undersat'
        }
        return reservoir_types.get(reservoir_name, 'Saturated')
    
    def _classify_gor_trend(self, well_features, reservoir_name):
        """Classify GOR trend relative to solution GOR"""
        # Solution GOR from reservoir info
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
        
        # Select numerical features for ML
        feature_columns = [
            'Total_Oil_Prod', 'Total_Gas_Prod', 'Total_Water_Prod',
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
        
        # Create feature matrix
        X = self.processed_features[feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        print(f"Features prepared! Shape: {self.X_scaled.shape}")
        
    def train_gaussian_nb_model(self):
        """Train Gaussian Naive Bayes model"""
        print("\nTraining Gaussian Naive Bayes model...")
        
        # Prepare target variables
        y_encoded = {}
        
        for target_col in self.target_columns:
            le = LabelEncoder()
            y_encoded[target_col] = le.fit_transform(self.target_data[target_col].astype(str))
            self.label_encoders[target_col] = le
        
        # Create multi-output classifier
        self.model = MultiOutputClassifier(GaussianNB())
        
        # Prepare target matrix
        y_matrix = np.column_stack([y_encoded[col] for col in self.target_columns])
        
        # Train model
        self.model.fit(self.X_scaled, y_matrix)
        
        # Make predictions
        y_pred = self.model.predict(self.X_scaled)
        
        # Calculate accuracy for each target
        accuracies = {}
        for i, target_col in enumerate(self.target_columns):
            acc = accuracy_score(y_matrix[:, i], y_pred[:, i])
            accuracies[target_col] = acc
            print(f"{target_col}: {acc:.3f}")
        
        print(f"\nOverall accuracy: {np.mean(list(accuracies.values())):.3f}")
        
    def explain_gaussian_nb(self):
        """Explain Gaussian Naive Bayes model and parameters"""
        print("\n" + "="*50)
        print("GAUSSIAN NAIVE BAYES MODEL EXPLANATION")
        print("="*50)
        
        print("""
        GAUSSIAN NAIVE BAYES OVERVIEW:
        ============================
        
        Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem
        with the assumption of independence between features and that continuous
        features follow a normal (Gaussian) distribution.
        
        MATHEMATICAL FOUNDATION:
        ======================
        
        Bayes' Theorem: P(class|features) = P(features|class) * P(class) / P(features)
        
        For Gaussian NB:
        - P(class): Prior probability of each class
        - P(features|class): Likelihood assuming Gaussian distribution
        - P(xi|class) = (1/√(2π*σ²)) * exp(-(xi-μ)²/(2σ²))
        
        Where μ is the mean and σ is the standard deviation of feature xi for the given class.
        
        KEY ASSUMPTIONS:
        ==============
        1. Feature Independence: Features are conditionally independent given the class
        2. Gaussian Distribution: Continuous features follow normal distribution
        3. No missing values in prediction phase
        
        MODEL PARAMETERS:
        ===============
        """)
        
        # Display model parameters for each target
        for i, target_col in enumerate(self.target_columns):
            estimator = self.model.estimators_[i]
         
            print(f"\n{target_col}:")
            print(f"  - Classes: {self.label_encoders[target_col].classes_}")
            print(f"  - Class prior probabilities: {estimator.class_prior_}")
            print(f"  - Feature means shape: {estimator.theta_.shape}")
            print(f" - Feature variances shape: {estimator.var_.shape}")
        
        print(f"""
        
        ADVANTAGES FOR THIS PROBLEM:
        ==========================
        1. Works well with small datasets (20 wells)
        2. Fast training and prediction
        3. Handles multiple classes naturally
        4. Provides probabilistic outputs
        5. Robust to irrelevant features
        6. No hyperparameter tuning required
        
        LIMITATIONS:
        ==========
        1. Strong independence assumption
        2. Assumes Gaussian distribution for continuous features
        3. Can be affected by skewed distributions
        4. May not capture complex feature interactions
        
        FEATURE IMPORTANCE:
        =================
        Features with higher variance differences between classes
        are more discriminative for classification.
        """)
        
    def generate_classification_results(self):
        """Generate final classification results"""
        print("\nGenerating classification results...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_scaled)
        
        # Convert predictions back to original labels
        results = []
        for i in range(len(y_pred)):
            well_result = {'Well': i + 1}
            for j, target_col in enumerate(self.target_columns):
                predicted_class = self.label_encoders[target_col].inverse_transform([y_pred[i][j]])[0]
                well_result[target_col] = predicted_class
            results.append(well_result)
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv('Team_DSEATS_Africa_2025_Classification.csv', index=False)
        
        print("Classification results generated and saved!")
        print(results_df.head())
        
        return results_df
    
    def visualize_results(self):
        """Create visualizations for EDA and results"""
        print("\nCreating visualizations...")
        
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
                             'Total_Oil_Prod', 'Avg_PI']
        corr_matrix = self.processed_features[important_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6)
        ax6.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('well_classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional plot: Model Performance
        self._plot_model_performance()
        
    def _plot_model_performance(self):
        """Plot model performance metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Accuracy by Target Variable
        ax1 = axes[0]
        y_pred = self.model.predict(self.X_scaled)
        accuracies = []
        
        for i, target_col in enumerate(self.target_columns):
            y_true = self.target_data[target_col].astype(str)
            y_pred_decoded = self.label_encoders[target_col].inverse_transform(y_pred[:, i])
            acc = accuracy_score(y_true, y_pred_decoded)
            accuracies.append(acc)
        
        bars = ax1.bar(range(len(self.target_columns)), accuracies, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
        ax1.set_title('Model Accuracy by Target Variable', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Target Variables')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(self.target_columns)))
        ax1.set_xticklabels([col.replace(' ', '\n') for col in self.target_columns], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
    
        # Plot 2: Prediction Confidence
        ax2 = axes[1]
        # Get prediction probabilities for first target (Reservoir Name)
        proba = self.model.estimators_[0].predict_proba(self.X_scaled)
        max_proba = np.max(proba, axis=1)

        ax2.hist(max_proba, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Maximum Prediction Probability')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_complete_pipeline(self):
        """Run the complete pipeline from data loading to results generation"""
        print("="*60)
        print("SPE DSEATS AFRICA 2025 - WELL CLASSIFICATION PIPELINE")
        print("="*60)
            
        # Step 1: Load data
        self.load_data()
            
        # Step 2: Clean data
        self.clean_data()
            
        # Step 3: Feature engineering
        self.feature_engineering()
            
        # Step 4: Prepare target variables
        self.prepare_target_variables()
            
        # Step 5: Prepare features for ML
        self.prepare_features_for_ml()
            
        # Step 6: Train model
        self.train_gaussian_nb_model()
            
        # Step 7: Explain model
        self.explain_gaussian_nb()
            
        # Step 8: Generate results
        results = self.generate_classification_results()
            
        # Step 9: Create visualizations
        self.visualize_results()
            
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
            
        return results
        
    def cross_validate_model(self):
        """Perform cross-validation to assess model robustness"""
        print("\nPerforming cross-validation...")
            
        cv_scores = {}
            
        for i, target_col in enumerate(self.target_columns):
            # Get target values
            y_true = self.target_data[target_col].astype(str)
            y_encoded = self.label_encoders[target_col].transform(y_true)
                
            # Single-output classifier for cross-validation
            single_model = GaussianNB()
                
            # Perform 5-fold cross-validation
            scores = cross_val_score(single_model, self.X_scaled, y_encoded, 
                                cv=5, scoring='accuracy')
                
            cv_scores[target_col] = {
                 'mean': scores.mean(),
                 'std': scores.std(),
                 'scores': scores
              }
               
            print(f"{target_col}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            
        return cv_scores
        
    def generate_detailed_report(self):
        """Generate a detailed classification report"""
        print("\nGenerating detailed classification report...")
            
        # Make predictions
        y_pred = self.model.predict(self.X_scaled)
            
        # Generate report for each target
        reports = {}
            
        for i, target_col in enumerate(self.target_columns):
            y_true = self.target_data[target_col].astype(str)
            y_pred_decoded = self.label_encoders[target_col].inverse_transform(y_pred[:, i])
                
            # Classification report
            report = classification_report(y_true, y_pred_decoded, output_dict=True)
            reports[target_col] = report
                
            print(f"\n{target_col}:")
            print(classification_report(y_true, y_pred_decoded))
                
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred_decoded)
            print(f"Confusion Matrix:\n{cm}")
            
        return reports
        
    def save_model_artifacts(self):
         """Save model and preprocessing artifacts"""

        
            
            # Save model
         with open('gaussian_nb_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
                
            # Save scaler
         with open('feature_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
                
            # Save label encoders
         with open('label_encoders.pkl', 'wb') as f:
                pickle.dump(self.label_encoders, f)
                
            # Save processed features
         self.processed_features.to_csv('processed_features.csv', index=False)
                
            # Save target data
         self.target_data.to_csv('target_data.csv', index=False)
                
         print("Model artifacts saved successfully!")
            
    def load_model_artifacts(self):
        """Load saved model and preprocessing artifacts"""
        
            
        # Load model
        with open('gaussian_nb_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
            
        # Load scaler
        with open('feature_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load label encoders
        with open('label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
            
        print("Model artifacts loaded successfully!")
        
    def predict_new_well(self, well_features):
        """Predict classifications for a new well"""
        if self.model is None:
            raise ValueError("Model not trained. Run train_gaussian_nb_model() first.")
            
        # Scale features
        well_features_scaled = self.scaler.transform([well_features])
            
        # Make prediction
        prediction = self.model.predict(well_features_scaled)[0]
            
        # Get probabilities
        probabilities = {}
        for i, target_col in enumerate(self.target_columns):
            proba = self.model.estimators_[i].predict_proba(well_features_scaled)[0]
            probabilities[target_col] = dict(zip(
                self.label_encoders[target_col].classes_,
                proba
             ))
            
        # Decode predictions
        results = {}
        for i, target_col in enumerate(self.target_columns):
            predicted_class = self.label_encoders[target_col].inverse_transform([prediction[i]])[0]
            results[target_col] = {
               'prediction': predicted_class,
               'probabilities': probabilities[target_col]
            }
            
        return results

    # Example usage and execution
if __name__ == "__main__":
        # Initialize pipeline
        pipeline = WellClassificationPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Perform cross-validation
        cv_scores = pipeline.cross_validate_model()
        
        # Generate detailed report
        detailed_reports = pipeline.generate_detailed_report()
        
        # Save model artifacts
        pipeline.save_model_artifacts()
        
        print("\n" + "="*60)
        print("ADDITIONAL ANALYSIS COMPLETED!")
        print("="*60)
        
        # Example of predicting for a new well
        print("\nExample prediction for a hypothetical well:")
        example_features = [
            50000,  # Total_Oil_Prod
            150000,  # Total_Gas_Prod
            20000,  # Total_Water_Prod
            200,  # Avg_Daily_Oil
            500,  # Max_Daily_Oil
            -0.1,  # Oil_Production_Decline
            2800,  # Avg_BHP
            3000,  # Max_BHP
            2500,  # Min_BHP
            10000,  # BHP_Variance
            150,  # Avg_WHP
            50,  # Avg_Annulus_Press
            180,  # Avg_Downhole_Temp
            120,  # Avg_Wellhead_Temp
            45,  # Avg_Choke_Size
            100,  # Choke_Variance
            20,  # Avg_Onstream_Hours
            800,  # Avg_GOR
            0,  # GOR_Trend
            1200,  # Max_GOR
            30,  # Avg_Water_Cut
            1,  # Water_Cut_Trend
            60,  # Max_Water_Cut
            0.5,  # Avg_PI
            0,  # PI_Trend
            1.0,  # Max_PI
            0.7,  # Production_Stability
            300,  # Days_Produced
            0.8,  # Production_Efficiency
            2650,  # Pressure_Differential
            1  # Has_Annulus_Pressure
        ]
        
        try:
            predictions = pipeline.predict_new_well(example_features)
            print("\nPredictions for example well:")
            for target, result in predictions.items():
                print(f"{target}: {result['prediction']}")
        except Exception as e:
            print(f"Error in prediction: {e}")
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE!")
        print("Files generated:")
        print("- DataPhandas_DSEATS_Africa_2025_Classification.csv")
        print("- well_classification_analysis.png")
        print("- model_performance_analysis.png")
        print("- processed_features.csv")
        print("- target_data.csv")
        print("- gaussian_nb_model.pkl")
        print("- feature_scaler.pkl")
        print("- label_encoders.pkl")
        print("="*60)
        
        