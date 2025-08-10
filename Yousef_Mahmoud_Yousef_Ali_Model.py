import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
from sklearn.preprocessing import StandardScaler

wr.filterwarnings('ignore')

print("=" * 50)
print("LOADING AND EXPLORING THE DATASET")
print("=" * 50)

data_file = pd.read_csv("first inten project.csv")
data_file.columns = [col.strip() for col in data_file.columns]

print("Shape: ", data_file.shape)
print("First 5 rows:\n", data_file.head())
print("Data types:\n", data_file.dtypes)
print("Nulls:\n", data_file.isnull().sum())

before_data = data_file.copy()

print("\n" + "=" * 50)
print("OUTLIER DETECTION AND REMOVAL")
print("=" * 50)

before_outliers = before_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numerical columns: ", before_outliers)

def IQR_METHOD(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

plt.figure(figsize=(10, 5))
plt.title("Boxplot Before Outlier Removal")
plt.boxplot(before_data[before_outliers].values, labels=before_outliers, vert=False)
plt.show()

print("\nOutlier Removal:")
for col in before_outliers:
    before_rows = data_file.shape[0]
    data_file = IQR_METHOD(data_file, col)
    after_rows = data_file.shape[0]
    print(f"{col}: removed {before_rows - after_rows} outliers using IQR Method")

columns_to_drop = ['repeated', 'P-not-C', 'car parking space', 'number of week nights']
existing_columns = [col for col in columns_to_drop if col in data_file.columns]
data_file = data_file.drop(columns=existing_columns)

outliers1 = data_file.select_dtypes(include=['int64', 'float64']).columns.tolist()

plt.figure(figsize=(10, 5))
plt.title("Boxplot After Outlier Removal")
plt.boxplot(data_file[outliers1].values, labels=outliers1, vert=False)
plt.show()

print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.countplot(x="booking status", data=data_file)
plt.title("Booking Distribution")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data_file['lead time'], bins=30, kde=True)
plt.title("The lead time Dist.")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='type of meal', data=data_file, order=data_file["type of meal"].value_counts().index)
plt.title("Meal Types Distribution")
plt.xticks(rotation=14)
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data_file['average price'], bins=30, kde=True)
plt.title("The average price Dist.")
plt.xlabel("Average")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='special requests', data=data_file)
plt.title("Distribution of Special Requests")
plt.xlabel("Number of Requests")
plt.ylabel("Number of Bookings")
plt.show()

data_file['date of reservation'] = pd.to_datetime(data_file['date of reservation'], errors='coerce')
data_file['reservation_month'] = data_file['date of reservation'].dt.to_period("M")
plt.figure(figsize=(10, 5))
data_file["reservation_month"].value_counts().sort_index().plot()
plt.title("The Most Monthly Booking")
plt.xlabel("Month")
plt.ylabel("No. Of bookings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

cat_col = data_file.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns: ", cat_col)
for col in cat_col:
    data_file[col] = data_file[col].str.strip()

data_file['date of reservation'] = pd.to_datetime(data_file['date of reservation'], errors='coerce')
data_file['reservation_month'] = data_file['date of reservation'].dt.to_period("M")
data_file['reservation_day'] = data_file['date of reservation'].dt.day
data_file['reservation_weekday'] = data_file['date of reservation'].dt.dayofweek

data_file = data_file.dropna()
encode = LabelEncoder()
data_file['booking status'] = encode.fit_transform(data_file['booking status'])
with open("label_encoder.pkl" , "wb") as f:
    pickle.dump(encode , f)

data_file = pd.get_dummies(data_file, columns=['type of meal', 'room type', 'market segment type'], drop_first=True)

print("Shape after preprocessing: ", data_file.shape)
print("Columns after preprocessing: ", data_file.columns.tolist())

print("\n" + "=" * 50)
print("MACHINE LEARNING MODEL")
print("=" * 50)

X = data_file.drop(columns=['booking status', 'Booking_ID', 'date of reservation', 'reservation_month'])
Y = data_file['booking status']
with open("model_features.pkl" , "wb") as f:
    pickle.dump(X.columns.tolist() , f)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)
print("Training set: ", X_train.shape)
print("Test split: ", X_test.shape)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
with open("scaler.pkl","wb") as f:
    pickle.dump(sc , f)

Model = LogisticRegression(max_iter=2000)
Model.fit(X_train_scaled, Y_train)
pickle.dump(Model, open("Model.pkl","wb"))
y_predict = Model.predict(X_test_scaled)
Acc = accuracy_score(Y_test, y_predict)

print("\n" + "=" * 30)
print("MODEL PERFORMANCE RESULTS")
print("=" * 30)
print("Accuracy: ", round(Acc * 100, 2), "%")
print("\nClassification Report:\n", classification_report(Y_test, y_predict))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, y_predict))

ConfusionMatrixDisplay.from_estimator(Model, X_test, Y_test)
plt.title("Confusion Matrix")
plt.show()

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE!")
print("=" * 50)



