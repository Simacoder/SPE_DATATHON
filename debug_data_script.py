import pandas as pd
import os

print("=== DATA STRUCTURE DEBUGGING SCRIPT ===")

# List all CSV files in current directory and subdirectories
print("\n1. SEARCHING FOR CSV FILES:")
csv_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

if csv_files:
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")
else:
    print("No CSV files found in current directory or subdirectories")

print("\n2. EXAMINING EACH CSV FILE:")
for file in csv_files:
    try:
        print(f"\n=== {file} ===")
        df = pd.read_csv(file)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Show first few rows
        print("First 3 rows:")
        print(df.head(3))
        
        # Check for well-related columns
        well_cols = [col for col in df.columns if 'WELL' in col.upper() or 'NAME' in col.upper()]
        if well_cols:
            print(f"Potential well columns: {well_cols}")
            for col in well_cols:
                print(f"  {col}: {df[col].nunique()} unique values")
                print(f"  Sample values: {df[col].unique()[:5]}")
        
        # Check for date columns
        date_cols = [col for col in df.columns if 'DATE' in col.upper() or 'TIME' in col.upper()]
        if date_cols:
            print(f"Potential date columns: {date_cols}")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"Error reading {file}: {e}")

print("\n3. RECOMMENDATIONS:")
print("Based on the file structure analysis above:")
print("1. Update the file paths in the main script")
print("2. Update the column names based on actual column headers")
print("3. Make sure the well identifier column is consistent across files")