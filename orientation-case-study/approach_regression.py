import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Load the CSV file
df = pd.read_csv('creditcard_dataset.csv')

# Convert 'DEFAULT PAYMENT NEXT MONTH' to binary, handling potential NaN values
df['DEFAULT PAYMENT NEXT MONTH'] = df['DEFAULT PAYMENT NEXT MONTH'].apply(lambda x: 1 if isinstance(x, str) and x.lower() == 'yes' else 0 if isinstance(x, str) else x)

# Drop rows where the target column 'DEFAULT PAYMENT NEXT MONTH' is NaN
df = df.dropna(subset=['DEFAULT PAYMENT NEXT MONTH'])

# Prepare feature columns
features = df[['AMOUNT OF GIVEN CREDIT', 'GENDER', 'EDUCATION', 'MARRIAGE', 'AGE', 
               'PAYMENT STATUS AT T', 'PAYMENT STATUS AT T-1', 'PAYMENT STATUS AT T-2',
               'PAYMENT STATUS AT T-3', 'PAYMENT STATUS AT T-4', 'PAYMENT STATUS AT T-5',
               'BILL STATEMENT AT T ($)', 'BILL STATEMENT AT T-1 ($)', 'BILL STATEMENT AT T-2 ($)', 
               'BILL STATEMENT AT T-3 ($)', 'BILL STATEMENT AT T-4 ($)', 'BILL STATEMENT AT T-5 ($)', 
               'PAYMENT AT T ($)', 'PAYMENT AT T-1 ($)', 'PAYMENT AT T-2 ($)', 
               'PAYMENT AT T-3 ($)', 'PAYMENT AT T-4 ($)', 'PAYMENT AT T-5 ($)']]

# One-hot encode categorical columns
features = pd.get_dummies(features, columns=['GENDER', 'EDUCATION', 'MARRIAGE', 
                                             'PAYMENT STATUS AT T', 'PAYMENT STATUS AT T-1', 
                                             'PAYMENT STATUS AT T-2', 'PAYMENT STATUS AT T-3', 
                                             'PAYMENT STATUS AT T-4', 'PAYMENT STATUS AT T-5'])

# Handling missing values in features using SimpleImputer
imputer = SimpleImputer(strategy='mean')
features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

# Target column
target = df['DEFAULT PAYMENT NEXT MONTH']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
