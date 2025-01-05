import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Data (berdasarkan gambar)
data = pd.DataFrame({
    'Nama': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'Usia': [30, 32, 29, 35, 28, 31, 40, 27, 34, 26],
    'Jenis Kelamin': ['Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita'],
    'Tenure': [3, 5, 2, 8, 1, 3, 10, 2, 7, 1],
    'Jabatan': ['Staff', 'Supervisor', 'Staff', 'Manager', 'Analis', 'Staff', 'Direktur', 'Staff', 'Supervisor', 'Staff'],
    'Level Gaji': [3, 4, 3, 5, 2, 3, 6, 2, 5, 2],
    'Kampus': ['UI', 'ITB', 'UGM', 'ITS', 'Unpad', 'Binus', 'Trisakti', 'Mercu Buana', 'Pancasila', 'Atma Jaya'],
    'Turnover': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
})

# Encode categorical data
data['Jenis Kelamin'] = LabelEncoder().fit_transform(data['Jenis Kelamin'])
data['Jabatan'] = LabelEncoder().fit_transform(data['Jabatan'])
data['Kampus'] = LabelEncoder().fit_transform(data['Kampus'])

# Split data into features and target
X = data.drop(['Nama', 'Turnover'], axis=1)
y = data['Turnover']

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)