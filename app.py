from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model once
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        education = request.form['education']
        age = float(request.form['age'])
        experience = float(request.form['experience'])
        job_role = request.form['job_role']
        department = request.form['department']

        input_df = pd.DataFrame([[education, age, experience, job_role, department]],
                                columns=['Education', 'Age', 'Experience', 'JobRole', 'Department'])

        predicted_salary = model.predict(input_df)[0]
        return render_template('index.html', prediction_text=f'Estimated Salary for {name}: ₹{predicted_salary:,.2f}')
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting app on port {port}...")
    app.run(host='0.0.0.0', port=port)
