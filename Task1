# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
Data Collection and Exploration
# Assuming you have a CSV file with sales data, load it into a DataFrame
sales_data = pd.read_csv('sales_data.csv')

#Explore the dataset
print(sales_data.head())
print(sales_data.info())

#Data Preprocessing
# Assume 'advertising_expenditure', 'target_audience', 'platform_selection' are relevant features
X = sales_data[['advertising_expenditure', 'target_audience', 'platform_selection']]
y = sales_data['sales']

# Handle missing values, outliers, and perform any necessary data cleaning here

# Convert categorical variables into numerical representations (if needed)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Selection
# Skip for simplicity, but you may want to explore feature selection techniques

# Model Selection
# Choose a linear regression model
model = LinearRegression()

#Model Training
model.fit(X_train, y_train)

#Model Evaluation
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

#Prediction and Optimization
# Use the trained model to make predictions on new data

#Visualization
# Visualize predictions vs. actual sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.show()