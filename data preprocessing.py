
import pandas as pd
import numpy as np


df = pd.read_csv("notebook/aqi data.csv")

#EDA on dataset
print("\nData Types & Null Info:")
print(df.info())

total_nulls = df.isnull().sum().sum()

print("Initial Shape:", df.shape)
print("\nTotal null values in dataset:", total_nulls)
# Count null values in each column
null_counts = df.isnull().sum()

print("\nNull values per column:")
print(null_counts)


# Select only numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Fill null values with column mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print("\nNull values after mean imputation:")
print(df[numeric_cols].isnull().sum())


print("\nTotal null values in dataset after filling:")
print(df.isnull().sum().sum())


print("\nComplete Dataset Summary (Numeric + Categorical):")
print(df.describe(include="all"))




# STANDARDIZE COLUMN NAMES
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "m_")


# HANDLE INVALID PLACEHOLDERS
invalid_values = ["", " ", "unknown", "Unknown", "N/A", "na", "null", -999, 999999]
df.replace(invalid_values, np.nan, inplace=True)


# HANDLE DATA TYPES
# Convert numeric-looking columns safely
numeric_cols = [
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed",
    "rainfall", "aqi"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# HANDLE MISSING VALUES
# Fill numeric columns with mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# REMOVE DUPLICATES
df.drop_duplicates(inplace=True)


# OUTLIER HANDLING (IQR METHOD)

for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

# FIX FLOAT PRECISION (3 DECIMALS)

float_cols = df.select_dtypes(include=["float64"]).columns
df[float_cols] = df[float_cols].round(3)


# FINAL VALIDATION

print("\nFinal Shape:", df.shape)
print("Remaining Null Values:", df.isnull().sum().sum())


numeric_cols = df.select_dtypes(include=["int64", "float64"])

numeric_summary = numeric_cols.describe()

print("\nFinal Numeric Statistics (No NaNs):")
print(numeric_summary)

# 10. SAVE CLEANED DATA
output_path = "notebook/aqi_cleaned_processed.csv"
df.to_csv(output_path, index=False)

print("\nCleaned & processed data saved to:", output_path)


