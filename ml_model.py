import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv("data.csv")

selected_columns = [
    'Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
    'Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date',
    'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries',
    'Outstanding_Debt','Credit_Utilization_Ratio','Credit_History_Age',
    'Payment_of_Min_Amount','Total_EMI_per_month','Amount_invested_monthly',
    'Monthly_Balance'
]

X = df[selected_columns]
y = df["Credit_Score"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as model.pkl")
