import pandas as pd
import numpy as np

def load_data(file1, file2):
    # Load the datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    return df1, df2

def unify_columns(df1, df2, mapping1=None, mapping2=None):
    # Rename columns to match
    if mapping1 is not None:
        df1.rename(columns=mapping1, inplace=True)
    if mapping2 is not None:
        df2.rename(columns=mapping2, inplace=True)
    print('Columns renamed successfully!')

    # Drop columns that are not in both datasets
    columns_to_drop = set(df1.columns) ^ set(df2.columns)
    for column in columns_to_drop:
        if column in df1.columns:
            df1.drop(columns=column, inplace=True)
        if column in df2.columns:
            df2.drop(columns=column, inplace=True)
    print('Columns that are not in both datasets dropped successfully!')

    return df1, df2

def standardize_formats(df):
    # Lower case and replace spaces with underscores in object columns
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.lower()
            df[column] = df[column].str.replace(' ', '_')

    return df

def standardize_date(df, date_column, date_format):
    # Convert to datetime using the specified format
    df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
    return df

def standardize_phone(df, phone_column):
    # Remove non-numeric characters
    df[phone_column] = df[phone_column].str.replace(r'\D', '', regex=True)
    return df

def drop_duplicates(df):
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def handle_missing_values(df, fill_value):
    # Fill rows with missing values
    df.fillna(fill_value, inplace=True)
    return df

## Constants for column mappings
ID = 'ID'
NAME = 'Name'
EMAIL = 'Email'
PHONE = 'Phone'
ADDRESS = 'Address'
DOB = 'DOB'
TRANSACTION_A = 'Transaction_Amount'
TRANSACTION_D = 'Transaction_Date'

def main():
    file1 = './matching_techniques/datasets/Base_A.csv'
    file2 = './matching_techniques/datasets/Base_B.csv'

    df1, df2 = load_data(file1, file2)

    # Handle missing values
    fill_value = 'null'
    df1 = handle_missing_values(df1, fill_value)
    df2 = handle_missing_values(df2, fill_value)
    print('Missing values handled successfully!')

    # Unify columns



    mapping1 = {
        'Customer_ID': ID, 'Name': NAME, 'Email': EMAIL, 'Phone': PHONE, 'Address': ADDRESS, 'Date_of_Birth': DOB, 'Transaction_Amount': TRANSACTION_A, 'Transaction_Date': TRANSACTION_D
    }

    mapping2 = {
        'Client_ID': ID, 'Full_Name': NAME, 'Email_Address': EMAIL, 'Contact_No': PHONE, 'Residence': ADDRESS, 'DOB': DOB,  'Transaction_Amount': TRANSACTION_A, 'Transaction_Date': TRANSACTION_D
    }

    df1, df2 = unify_columns(df1, df2, mapping1, mapping2)
    print('Columns unified successfully!')

    # Standardize formats
    df1 = standardize_formats(df1)
    df2 = standardize_formats(df2)
    print('Formats standardized successfully!')

    # Standardize date formats
    df1 = standardize_date(df1, DOB, '%Y-%m-%d')
    df2 = standardize_date(df2, DOB, '%d/%m/%Y')
    print('Dates standardized successfully!')

    df1 = standardize_date(df1, TRANSACTION_D, '%Y-%m-%d')
    df2 = standardize_date(df2, TRANSACTION_D, '%d/%m/%Y')
    print('LPD standardized successfully!')

    # Standardize phone numbers
    df1 = standardize_phone(df1, PHONE)
    df2 = standardize_phone(df2, PHONE)
    print('Phone numbers standardized successfully!')

    # Drop duplicates
    df1 = drop_duplicates(df1)
    df2 = drop_duplicates(df2)
    print('Duplicates dropped successfully!')

    # Save the cleaned datasets
    df1.to_csv('BaseA_cleaned.csv', index=False, sep=',')
    df2.to_csv('BaseB_cleaned.csv', index=False, sep=',')
    print('Datasets cleaned and saved successfully!')

if __name__ == "__main__":
    main()
