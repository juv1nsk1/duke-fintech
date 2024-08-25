import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('creditcard_dataset.csv')

# Preview the data
print(data.head())

# Data Preprocessing: Convert categorical variables to numerical values
# Example conversions (need to adjust based on actual data in the CSV)
data['GENDER'] = data['GENDER'].map({'male': 1, 'female': 0})
data['EDUCATION'] = data['EDUCATION'].map({'graduate school': 1, 'university': 2, 'high school': 3, 'others': 4})
data['MARRIAGE'] = data['MARRIAGE'].map({'married': 1, 'single': 2, 'others': 3})
data['DEFAULT PAYMENT NEXT MONTH'] = data['DEFAULT PAYMENT NEXT MONTH'].map({'yes': 1, 'no': 0})

# Check for missing values and handle them if necessary
print(data.isnull().sum())

# Exploratory Data Analysis (EDA)

# Relationship between 'AMOUNT OF GIVEN CREDIT' and 'DEFAULT PAYMENT NEXT MONTH'
plt.figure(figsize=(10, 5))
sns.boxplot(x='DEFAULT PAYMENT NEXT MONTH', y='AMOUNT OF GIVEN CREDIT', data=data)
plt.title('Amount of Given Credit vs Default Payment Next Month')
plt.show()

# Relationship between 'GENDER' and 'DEFAULT PAYMENT NEXT MONTH'
plt.figure(figsize=(5, 5))
sns.countplot(x='GENDER', hue='DEFAULT PAYMENT NEXT MONTH', data=data)
plt.title('Gender vs Default Payment Next Month')
plt.show()

# Relationship between 'EDUCATION' and 'DEFAULT PAYMENT NEXT MONTH'
plt.figure(figsize=(8, 5))
sns.countplot(x='EDUCATION', hue='DEFAULT PAYMENT NEXT MONTH', data=data)
plt.title('Education vs Default Payment Next Month')
plt.show()

# Relationship between 'MARRIAGE' and 'DEFAULT PAYMENT NEXT MONTH'
plt.figure(figsize=(5, 5))
sns.countplot(x='MARRIAGE', hue='DEFAULT PAYMENT NEXT MONTH', data=data)
plt.title('Marriage vs Default Payment Next Month')
plt.show()

# Relationship between 'AGE' and 'DEFAULT PAYMENT NEXT MONTH'
plt.figure(figsize=(10, 5))
sns.boxplot(x='DEFAULT PAYMENT NEXT MONTH', y='AGE', data=data)
plt.title('Age vs Default Payment Next Month')
plt.show()
