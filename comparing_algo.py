# Path for the Train Dataset -> Z:\sem 5\ML package\Dataset\Dataset\
    
# Importing Data 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # Import the RandomForestClassifier for classification tasks
from sklearn.ensemble import RandomForestRegressor  # Use this for regression tasks

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Now you can use y_train_encoded as the target variable for XGBoost


adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)  # You can adjust the number of estimators


# Initialize the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of trees


train_data = pd.read_csv(r'C:\Users\smrit\OneDrive\Desktop\ML package\Dataset\Dataset\train\Largetrain.csv')
test_data = pd.read_csv(r'C:\Users\smrit\OneDrive\Desktop\ML package\Dataset\Dataset\test\Largetest.csv')
sorted_test_id = pd.read_csv(r'C:\Users\smrit\OneDrive\Desktop\ML package\Dataset\Dataset\sorted_test_id.csv')
train_labels = pd.read_csv(r'C:\Users\smrit\OneDrive\Desktop\ML package\Dataset\Dataset\trainlabels.csv')

print(train_data)
print(test_data)
print(sorted_test_id)
print(train_labels)

#Printing Dataset's Features, Sparsity level and Statistical Analysis
print("\t\t DATA FEATURES \t\t")
print("FEATURES:\n")
print("TRAIN DATA")
print(train_data.head())
print("-----------------------------------------------")
print(train_data.describe())
print("-----------------------------------------------")
print(train_data.info())

print("************************************************")
print("TEST DATA")
print(test_data.head())
print("-----------------------------------------------")
print(test_data.describe())
print("-----------------------------------------------")
print(test_data.info())

print("************************************************")
print("SORTED TEST ID")
print(sorted_test_id.head())
print("-----------------------------------------------")
print(sorted_test_id.describe())
print("-----------------------------------------------")
print(sorted_test_id.info())


print("************************************************")
print("TRAIN LABELS")
print(train_labels.head())
print("-----------------------------------------------")
print(train_labels.describe())
print("-----------------------------------------------")
print(train_labels.info())
print("\n")

print("SPARSITIY LEVEL:\n")
# Calculate the percentage of missing values for each feature
print("TRAIN DATA")
missing_percentage1 = (test_data.isnull().sum() / len(test_data)) * 100
print(missing_percentage1)

print("************************************************")
print("SORTED TEST ID")
missing_percentage2 = (sorted_test_id.isnull().sum() / len(sorted_test_id)) * 100
print(missing_percentage2)
print("************************************************")
print("TRAIN LABELS")
missing_percentage3 = (train_labels.isnull().sum() / len(train_labels)) * 100
print(missing_percentage3)
print("************************************************")


print("STATISTICAL ANALYSIS:\n")
print("TRAIN DATA")
train_data['Virtual'].hist()
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Virtual')
plt.show()
mean_value = train_data['Virtual'].mean()
mode_value = train_data['Virtual'].mode().values[0]  # In case there are multiple modes
median_value = train_data['Virtual'].median()

# Print the results
print(f"Mean: {mean_value}")
print(f"Mode: {mode_value}")
print(f"Median: {median_value}")
print("************************************************")
print("SORTED TEST ID")
print("It could not be ploted as the file contains only character values")
print("************************************************")
print("TRAIN LABELS")
train_labels['Class'].hist()
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Class')
plt.show()
print("************************************************")

X = train_data.drop(columns=['Class'])
y = train_data['Class']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''preprocessing data'''

# Define an imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on your training data
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

class_counts = y_train.value_counts()
print(class_counts)


''' preprocessing data over'''

print("\t\t++++++RANDOM FOREST ALGORITHM++++++\n")

# Initialize the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Convert 'target_names' to strings
target_names = [str(cn) for cn in train_data['Class'].unique()]

# Generate a classification report with 'Class' column included
classification_rep = classification_report(y_test, y_pred, target_names=target_names)
print(f'Classification Report:\n{classification_rep}')
confusion = confusion_matrix(y_test, y_pred)
# Print the confusion matrix and other metrics
print("Confusion Matrix:")
print(confusion)


print("\t\t++++++ADABOOST+++++++\n")

# Apply SMOTE to balance the class distribution
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize the Random Forest model
rf_classifier = AdaBoostClassifier(n_estimators=100, random_state=42)


# Fit the model to the training data
rf_classifier.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'AdaBoost Accuracy: {accuracy}')

# Convert 'target_names' to strings
target_names = [str(cn) for cn in train_data['Class'].unique()]

# Generate a classification report with 'Class' column included
classification_rep = classification_report(y_test, y_pred, target_names=target_names, zero_division=1)

print(f'AdaBoost Classification Report:\n{classification_rep}')
confusion = confusion_matrix(y_test, y_pred)
# Print the confusion matrix and other metrics
print("Confusion Matrix:")
print(confusion)
print("\t\t++++++XGBOOST++++++\n")

# Fit and transform the labels
y_train_encoded = label_encoder.fit_transform(y_train)


# Initialize the XGBoost model
xgb_classifier = XGBClassifier(verbosity=0)  # You can adjust verbosity as needed

# Fit the model to the training data
xgb_classifier.fit(X_train, y_train_encoded)  # Use y_train_encoded

# Make predictions on the test set
y_pred = xgb_classifier.predict(X_test)

# In case you want to decode the predictions to the original class labels:
y_pred_original = label_encoder.inverse_transform(y_pred)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Generate a classification report with original class labels
classification_rep = classification_report(y_test, y_pred_original)
print(f'Classification Report:\n{classification_rep}')
confusion = confusion_matrix(y_test, y_pred)
# Print the confusion matrix and other metrics
print("Confusion Matrix:")
print(confusion)

print("********************************************************")


