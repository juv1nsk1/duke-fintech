import pandas as pd  # Import the pandas library for data manipulation

# Open the CSV file and load the data into a DataFrame called 'df'
df = pd.read_csv('creditcard_dataset.csv')

# Count the total number of customers in the DataFrame
total_customer = len(df)

# Create a group of customers who have a "Default" payment status for the next month
group_default = df[df['DEFAULT PAYMENT NEXT MONTH'].str.contains('Yes', case=False, na=False)]

# Remove these customers from the original DataFrame
df = df[~df.index.isin(group_default.index)]

# Create a group of customers who had delays in any of the last 6 months
group_delayed = df[(df['PAYMENT STATUS AT T'].str.contains('delayed', case=False, na=False)) |
                   (df['PAYMENT STATUS AT T-1'].str.contains('delayed', case=False, na=False)) |
                   (df['PAYMENT STATUS AT T-2'].str.contains('delayed', case=False, na=False)) |
                   (df['PAYMENT STATUS AT T-3'].str.contains('delayed', case=False, na=False)) |
                   (df['PAYMENT STATUS AT T-4'].str.contains('delayed', case=False, na=False)) |
                   (df['PAYMENT STATUS AT T-5'].str.contains('delayed', case=False, na=False))]

# Remove these customers from the original DataFrame
df = df[~df.index.isin(group_delayed.index)]

# Create a group of customers who paid on time every month in the last 6 months
group_duly = df[(df['PAYMENT STATUS AT T'].str.contains('duly', case=False, na=False)) & 
                (df['PAYMENT STATUS AT T-1'].str.contains('duly', case=False, na=False)) & 
                (df['PAYMENT STATUS AT T-2'].str.contains('duly', case=False, na=False)) &
                (df['PAYMENT STATUS AT T-3'].str.contains('duly', case=False, na=False)) &
                (df['PAYMENT STATUS AT T-4'].str.contains('duly', case=False, na=False)) &
                (df['PAYMENT STATUS AT T-5'].str.contains('duly', case=False, na=False))]

# Count the number of customers in each group
num_default = len(group_default)
num_delayed = len(group_delayed)
num_duly = len(group_duly)

# Calculate the percentage of customers in each group relative to the total number of customers
percent_default = (num_default / total_customer) * 100
percent_delayed = (num_delayed / total_customer) * 100
percent_duly = (num_duly / total_customer) * 100

# Display the number and percentage of customers in each group
print(f"Group default: {num_default} ({percent_default:.2f}%)")
print(f"Group delayed: {num_delayed} ({percent_delayed:.2f}%)")
print(f"Group duly: {num_duly} ({percent_duly:.2f}%)")
