import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataframe = pd.read_excel('C:\\Users\\hello\\Python\\Student_Tests_Proj\\gradePrediction.xlsx',sheet_name=None)
data = dataframe['Training Data']

print(data.head())
print(len(data))
print(data.shape)
print(data.tail())
print(data.describe())

train_data = data.fillna(data.median())
train_data.columns = train_data.columns.astype(str)

#train_data = train_data.apply(lambda x: x[(x >= 0) & (x <= 100)]) 
train_data.iloc[:, :12] = train_data.iloc[:, :12].apply(lambda x: x[(x >= 0) & (x <= 100)])

train_data.fillna(train_data.median(), inplace=True) 
imputer = SimpleImputer(strategy='median')  # Impute missing values with median
train_data_cleaned = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)

print(train_data_cleaned.describe())

# Prepare features and target variable
X = train_data_cleaned.drop('FinalExam', axis=1)  # or other columns if necessary
y = train_data_cleaned['FinalExam']

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Calculate error between model predicted vs extsting value in valdiation set
mse = mean_squared_error(y_val, y_pred)
print("Mean Squared Error:", mse)

# Load and prepare test data similar to training data
test_data = dataframe['Testing Data']
testimputer = SimpleImputer(strategy='median') 
test_data.columns = test_data.columns.astype(str)
test_data_cleaned = pd.DataFrame(testimputer.fit_transform(test_data), columns=test_data.columns)
#test_data_cleaned = pd.DataFrame(testimputer.transform(test_data.drop('Year', axis=1)), columns=test_data.columns[:-1])

# Predict unknown final exam scores
final_exam_predictions = model.predict(test_data_cleaned)
test_data_cleaned['FinalExam'] = final_exam_predictions

print("new Testing data with predicted final exam scores\n")
print(test_data_cleaned)

# Lastly, save the DF with predicted final exam scores to an Excel file
test_data_cleaned.to_excel('TestingDataFinalsPredictions.xlsx', index=False, sheet_name='PredictedScores')

""" # Correlation heatmap
matplotlib.use('TkAgg') 
plt.figure(figsize=(10, 8))
correlation_matrix = train_data_cleaned.corr()  # Compute the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('heatmap.png') 
plt.show() """

