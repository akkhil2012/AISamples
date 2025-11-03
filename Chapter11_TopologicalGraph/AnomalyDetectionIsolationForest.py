import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')


class LogAnomalyDetector:
    """
    Log Anomaly Detection using Isolation Forest Algorithm

    This class implements anomaly detection in log data using Isolation Forest
    with configurable contamination rate, handling both numeric and string features
    without requiring explicit rule definitions.
    """

    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the anomaly detector

        Parameters:
        contamination (float): The proportion of outliers in the dataset (default: 0.1 for 10%)
        random_state (int): Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = None
        self.preprocessor = None
        self.feature_names = None
        self.label_encoders = {}

    def _preprocess_features(self, df, selected_features):
        """
        Preprocess features handling both numeric and string types

        Parameters:
        df (DataFrame): Input dataframe
        selected_features (list): List of feature column names to use

        Returns:
        numpy.ndarray: Preprocessed feature matrix
        """
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Convert potential numeric columns that might be read as strings
        for feature in selected_features:
            if feature in df.columns:
                # Try to convert to numeric, if it fails, leave as is
                df[feature] = pd.to_numeric(df[feature], errors='ignore')

        # Separate numeric and categorical features
        numeric_features = []
        categorical_features = []

        for feature in selected_features:
            if feature in df.columns:  # Only process features that exist
                if pd.api.types.is_numeric_dtype(df[feature]):
                    numeric_features.append(feature)
                else:
                    categorical_features.append(feature)

        # Create preprocessing pipeline
        preprocessors = []

        if numeric_features:
            preprocessors.append(('num', StandardScaler(), numeric_features))

        if categorical_features:
            preprocessors.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                                categorical_features))

        if not preprocessors:
            raise ValueError("No valid features selected for anomaly detection")

        self.preprocessor = ColumnTransformer(
            transformers=preprocessors,
            remainder='drop'
        )

        # Fit and transform the data
        try:
            X_processed = self.preprocessor.fit_transform(df)
            return X_processed
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            print(f"Numeric features: {numeric_features}")
            print(f"Categorical features: {categorical_features}")
            print(f"Data types: {df[selected_features].dtypes}")
            raise

    def fit_predict(self, full_path, selected_features):
        """
        Fit the Isolation Forest model and predict anomalies

        Parameters:
        df (DataFrame): Input dataframe containing log data
        selected_features (list): List of feature column names to use for anomaly detection

        Returns:
        tuple: Array of predictions (-1 for anomalies, 1 for normal) and anomaly scores
        """
        df = pd.read_csv(full_path)
        print(f"Processing {len(df)} log entries...")
        print(f"Selected features: {selected_features}")
        print(f"Contamination rate: {self.contamination * 100}%")

        # Preprocess features
        X_processed = self._preprocess_features(df, selected_features)

        # Initialize and fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )


        # Fit and predict
        predictions = self.isolation_forest.fit_predict(X_processed)

        # Calculate anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(X_processed)

        return predictions, anomaly_scores

    def analyze_results(self, df, predictions, anomaly_scores, selected_features):
        """
        Analyze and display results of anomaly detection

        Parameters:
        df (DataFrame): Original dataframe
        predictions (array): Prediction results from Isolation Forest
        anomaly_scores (array): Anomaly scores
        selected_features (list): Features used for detection
        """
        # Add results to dataframe
        results_df = df.copy()
        results_df['anomaly'] = predictions
        results_df['anomaly_score'] = anomaly_scores
        results_df['is_anomaly'] = predictions == -1

        # Summary statistics
        n_anomalies = sum(predictions == -1)
        n_normal = sum(predictions == 1)

        print("\n" + "="*50)
        print("ANOMALY DETECTION RESULTS")
        print("="*50)
        print(f"Total log entries: {len(df)}")
        print(f"Normal entries: {n_normal} ({n_normal/len(df)*100:.2f}%)")
        print(f"Anomalous entries: {n_anomalies} ({n_anomalies/len(df)*100:.2f}%)")
        
        if n_anomalies > 0:
            print(f"Detection threshold (anomaly score): {min(anomaly_scores[predictions == -1]):.4f}")

        # Display anomalous entries
        if n_anomalies > 0:
            print("\n" + "-"*50)
            print("DETECTED ANOMALIES (Top 10 by anomaly score):")
            print("-"*50)
            anomalies = results_df[results_df['is_anomaly'] == True].sort_values('anomaly_score')

            for idx, row in anomalies.head(10).iterrows():
                print(f"\nEntry {idx}:")
                for feature in selected_features:
                    print(f"  {feature}: {row[feature]}")
                print(f"  Anomaly Score: {row['anomaly_score']:.4f}")
        else:
            print("\nNo anomalies detected with current contamination rate.")
            print("Consider adjusting the contamination parameter if you expect anomalies.")

        return results_df

    def save_results(self, results_df, output_file="anomaly_results.csv"):
        """
        Save results to CSV file

        Parameters:
        results_df (DataFrame): Results dataframe
        output_file (str): Output file path
        """
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")


def load_csv_data(file_path):
    """
    Load log data from CSV file with comprehensive error handling
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded dataframe
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        if not file_path.lower().endswith('.csv'):
            print(f"Warning: File doesn't have .csv extension: {file_path}")
        
        # Try to load the CSV with different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Could not load CSV file with any standard encoding")
        
        # Basic validation
        if df.empty:
            raise ValueError("CSV file is empty")
        
        print(f"Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the file path and try again.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty or contains no data.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        print("Please check if the file is a valid CSV format.")
        return None
    except Exception as e:
        print(f"Unexpected error loading CSV: {e}")
        return None


def explore_data(df):
    """
    Display basic information about the dataset to help with feature selection
    
    Parameters:
    df (DataFrame): Input dataframe
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    
    print("\n" + "-"*30)
    print("COLUMN INFORMATION")
    print("-"*30)
    
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        print(f"{i:2d}. {col}")
        print(f"    Type: {dtype}")
        print(f"    Null values: {null_count} ({null_count/len(df)*100:.1f}%)")
        print(f"    Unique values: {unique_count}")
        
        # Show sample values for categorical columns
        if dtype == 'object' and unique_count <= 20:
            sample_values = df[col].value_counts().head(5)
            print(f"    Top values: {dict(sample_values)}")
        elif dtype in ['int64', 'float64']:
            print(f"    Range: {df[col].min():.2f} to {df[col].max():.2f}")
        print()


def get_user_feature_selection(df):
    """
    Interactive feature selection based on user input
    
    Parameters:
    df (DataFrame): Input dataframe
    
    Returns:
    list: Selected features for anomaly detection
    """
    print("\n" + "="*50)
    print("FEATURE SELECTION")
    print("="*50)
    
    columns = list(df.columns)
    
    print("Available columns:")
    for i, col in enumerate(columns, 1):
        print(f"{i:2d}. {col} ({df[col].dtype})")
    
    print("\nSelect features for anomaly detection:")
    print("Enter column numbers separated by commas (e.g., 1,3,5)")
    print("Or enter column names separated by commas (e.g., cpu_usage,memory_usage)")
    
    while True:
        try:
            user_input = input("\nYour selection: ").strip()
            
            if not user_input:
                print("Please enter your selection.")
                continue
            
            selected_features = []
            
            # Check if input contains numbers or column names
            if user_input.replace(',', '').replace(' ', '').isdigit():
                # Input contains numbers
                indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                for idx in indices:
                    if 0 <= idx < len(columns):
                        selected_features.append(columns[idx])
                    else:
                        print(f"Invalid column number: {idx + 1}")
                        raise ValueError("Invalid selection")
            else:
                # Input contains column names
                feature_names = [x.strip() for x in user_input.split(',')]
                for name in feature_names:
                    if name in columns:
                        selected_features.append(name)
                    else:
                        print(f"Column '{name}' not found in dataset")
                        raise ValueError("Invalid selection")
            
            if not selected_features:
                print("No valid features selected. Please try again.")
                continue
            
            print(f"\nSelected features: {selected_features}")
            confirm = input("Confirm selection? (y/n): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                return selected_features
            
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return None


def main():
    """
    Main function for CSV-based anomaly detection
    """
    print("Log Anomaly Detection using Isolation Forest")
    print("=" * 50)
    print("This tool analyzes CSV files for anomalies using machine learning")
    
    # Get file path from user
    while True:
        file_path = input("\nEnter the path to your CSV file: ").strip()
        
        if not file_path:
            print("Please enter a valid file path.")
            continue
        
        # Remove quotes if present
        file_path = file_path.strip('"\'')
        
        # Load data
        print(f"\nLoading data from: {file_path}")
        df = load_csv_data(file_path)
        
        if df is not None:
            break
        else:
            retry = input("Would you like to try another file? (y/n): ").strip().lower()
            if retry not in ['y', 'yes']:
                print("Exiting...")
                return None
    
    # Explore data
    explore_data(df)
    
    # Get feature selection from user
    selected_features = get_user_feature_selection(df)
    
    if selected_features is None:
        print("Feature selection cancelled. Exiting...")
        return None
    
    # Get contamination rate
    while True:
        try:
            contamination_input = input(f"\nEnter contamination rate (0.01-0.5, default 0.1): ").strip()
            if not contamination_input:
                contamination = 0.1
                break
            else:
                contamination = float(contamination_input)
                if 0.01 <= contamination <= 0.5:
                    break
                else:
                    print("Contamination rate must be between 0.01 and 0.5")
        except ValueError:
            print("Please enter a valid number.")
    
    # Initialize detector
    detector = LogAnomalyDetector(contamination=contamination)
    
    # Perform anomaly detection
    print(f"\nStarting anomaly detection...")
    predictions, anomaly_scores = detector.fit_predict(file_path, selected_features)
    
    # Analyze results
    results_df = detector.analyze_results(df, predictions, anomaly_scores, selected_features)
    
    # Save results
    output_file = input(f"\nEnter output filename (default: anomaly_results.csv): ").strip()
    if not output_file:
        output_file = "anomaly_results.csv"
    
    detector.save_results(results_df, output_file)
    
    return results_df


def load_sample_log_data():
    """
    Generate sample log data for demonstration (kept for backward compatibility)
    """
    print("Generating sample data for demonstration...")
    np.random.seed(42)
    n_samples = 1000

    # Generate sample log data
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'log_level': np.random.choice(['INFO', 'WARNING', 'ERROR', 'DEBUG'], n_samples, p=[0.7, 0.15, 0.10, 0.05]),
        'response_time': np.random.lognormal(mean=2, sigma=0.5, size=n_samples),
        'cpu_usage': np.random.normal(50, 15, n_samples),
        'memory_usage': np.random.normal(60, 20, n_samples),
        'user_agent': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge', 'Other'], n_samples),
        'status_code': np.random.choice([200, 301, 400, 404, 500], n_samples, p=[0.8, 0.05, 0.05, 0.05, 0.05]),
        'request_size': np.random.exponential(1000, n_samples)
    }

    # Add some deliberate anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)

    for idx in anomaly_indices:
        # Create anomalous values
        data['response_time'][idx] = np.random.uniform(50, 100)  # Very high response time
        data['cpu_usage'][idx] = np.random.uniform(95, 100)     # Very high CPU
        data['memory_usage'][idx] = np.random.uniform(95, 100)  # Very high memory

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage with CSV file upload
    print("Choose mode:")
    print("1. Analyze your CSV file")
    print("2. Run with sample data (demo)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            results = main()
            break
        elif choice == "2":
            print("\nRunning demo with sample data...")
            df = load_sample_log_data()
            selected_features = ['response_time', 'cpu_usage', 'memory_usage', 'log_level', 'status_code']
            detector = LogAnomalyDetector(contamination=0.1)
            predictions, anomaly_scores = detector.fit_predict(df, selected_features)
            results = detector.analyze_results(df, predictions, anomaly_scores, selected_features)
            detector.save_results(results)
            break
        else:
            print("Please enter 1 or 2.")
    
    if 'results' in locals() and results is not None:
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print("Thank you for using Log Anomaly Detector!")