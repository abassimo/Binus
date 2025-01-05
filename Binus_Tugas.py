# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Create sample data
data = pd.DataFrame({
    'Nama': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'Usia': [30, 32, 29, 35, 28, 31, 40, 27, 34, 26],
    'Jenis Kelamin': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # 0: Pria, 1: Wanita
    'Tenure (tahun)': [3, 5, 2, 8, 1, 3, 10, 2, 7, 1],
    'Jabatan': [0, 1, 0, 2, 3, 0, 4, 0, 1, 0],  # Encoded positions
    'Level Gaji': [3, 4, 3, 5, 2, 3, 6, 2, 5, 2],
    'Universitas': [
        'Universitas Indonesia', 'Institut Teknologi Bandung', 
        'Universitas Gadjah Mada', 'Institut Teknologi Sepuluh Nopember', 
        'Universitas Padjadjaran', 'Universitas Bina Nusantara', 
        'Universitas Trisakti', 'Universitas Mercu Buana', 
        'Universitas Pancasila', 'Universitas Atma Jaya'
    ],
    'Turnover': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1]  # 0: Tidak resign, 1: Resign
})

# Drop the 'Nama' and 'Universitas' columns for modeling
X = data.drop(['Nama', 'Universitas', 'Turnover'], axis=1)
y = data['Turnover']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)