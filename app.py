from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

label_mapping = {0: "Poor", 1: "Good", 2: "Standard"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f"feature{i}"]) for i in range(1, 19)]

        column_names = [
            'Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
            'Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date',
            'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries',
            'Outstanding_Debt','Credit_Utilization_Ratio','Credit_History_Age',
            'Payment_of_Min_Amount','Total_EMI_per_month','Amount_invested_monthly',
            'Monthly_Balance'
        ]

        input_df = pd.DataFrame([features], columns=column_names)

        prediction = model.predict(input_df)[0]
        output = label_mapping[prediction]

        return render_template("index.html", prediction_text=f"Prediction: {output}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
