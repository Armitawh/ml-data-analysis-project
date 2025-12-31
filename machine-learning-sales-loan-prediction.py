import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
# Replace with your local path or relative path
df = pd.read_csv("raw_dataset.csv")

# ----------------------------
# Step 2: Data Cleaning
# ----------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)
numeric_df = df.select_dtypes(include=[np.number])  

# Remove outliers using IQR method
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Add synthetic defaults if needed
if 'Defaulted' in df.columns:
    if df['Defaulted'].nunique() == 1 and df['Defaulted'].iloc[0] == 0:
        n_defaults = max(1, int(0.05 * len(df)))
        default_indices = np.random.choice(df.index, n_defaults, replace=False)
        df.loc[default_indices, 'Defaulted'] = 1
        print(f"⚠️ Added {n_defaults} synthetic default(s) to allow training.")

# ----------------------------
# Step 3: Regression Model (Linear Regression for Sales)
# ----------------------------
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop(columns=['Sales'])
y = df_encoded['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

predictions = reg_model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Linear Regression Model MSE: {mse}")
print(f"Linear Regression Model R2 Score: {r2}")

# Save predictions
output_df = pd.DataFrame({
    'Actual_Sales': y_test,
    'Predicted_Sales': predictions
})
output_df.to_csv("predictions.csv", index=False)
output_df.to_json("predictions.json", orient='records', lines=True)
print("✅ Sales predictions saved!")

# Plot actual vs predicted sales
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Sales', marker='o')
plt.plot(predictions, label='Predicted Sales', marker='x')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Step 4: Classification Model (Random Forest for Loan Default)
# ----------------------------
feature_cols = ['Income', 'Loan_Amount', 'Credit_Score']  # update based on dataset
target_col = 'Defaulted'

X_class = df_encoded[feature_cols]
y_class = df_encoded[target_col]

# Check target distribution
print("Target distribution:\n", y_class.value_counts())

if y_class.nunique() < 2:
    print("⚠️ Target has only one class. Random Forest cannot train properly.")
    predictions_class = np.zeros(len(y_class), dtype=int)
    probabilities = np.zeros(len(y_class), dtype=float)
else:
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_c, y_train_c)

    predictions_class = rf_model.predict(X_test_c)
    probabilities = rf_model.predict_proba(X_test_c)[:, 1]

    accuracy = accuracy_score(y_test_c, predictions_class)
    conf_matrix = confusion_matrix(y_test_c, predictions_class)
    class_report = classification_report(y_test_c, predictions_class)

    print(f"Random Forest Accuracy: {accuracy}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Feature importance
    feature_importances = rf_model.feature_importances_
    plt.figure(figsize=(8,5))
    plt.barh(feature_cols, feature_importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.show()

# Save predictions
output_class_df = X_class.copy()
output_class_df['Actual_Default'] = y_class
output_class_df['Predicted_Default'] = predictions_class
output_class_df['Default_Probability'] = probabilities

output_class_df.to_csv("loan_predictions.csv", index=False)
output_class_df.to_json("loan_predictions.json", orient='records', lines=True)
print("✅ Loan default predictions saved!")
