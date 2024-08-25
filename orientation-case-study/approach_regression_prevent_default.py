import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the CSV file
df = pd.read_csv('creditcard_dataset.csv')

# Convert 'DEFAULT PAYMENT NEXT MONTH' to binary
df['DEFAULT PAYMENT NEXT MONTH'] = df['DEFAULT PAYMENT NEXT MONTH'].apply(
    lambda x: 1 if isinstance(x, str) and x.lower() == 'yes' else 0 if isinstance(x, str) else x
)

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

# Separate numerical and categorical columns
numeric_features = features.select_dtypes(include=['int64', 'float64'])
categorical_features = features.select_dtypes(include=['object'])

# Print data types for debugging
print("Categorical features data types:")
print(categorical_features.dtypes)

# Handle missing values for categorical columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
categorical_features_imputed = pd.DataFrame(imputer_categorical.fit_transform(categorical_features), columns=categorical_features.columns)

# One-hot encode categorical columns
categorical_features_encoded = pd.get_dummies(categorical_features_imputed, drop_first=True)

# Handle missing values for numeric columns
imputer_numeric = SimpleImputer(strategy='mean')
numeric_features_imputed = pd.DataFrame(imputer_numeric.fit_transform(numeric_features), columns=numeric_features.columns)

# Combine the imputed numeric and categorical features
features_imputed = pd.concat([numeric_features_imputed, categorical_features_encoded], axis=1)

# Standardize features
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features_imputed), columns=features_imputed.columns)

# Target column
target = df['DEFAULT PAYMENT NEXT MONTH']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
