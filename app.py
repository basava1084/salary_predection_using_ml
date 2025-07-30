from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        name = request.form['name']
        education = request.form['education']
        age = float(request.form['age'])
        experience = float(request.form['experience'])
        job_role = request.form['job_role']
        department = request.form['department']

        # Ensure column names match exactly as used in training
        input_df = pd.DataFrame([[education, age, experience, job_role, department]],
                                columns=['Education', 'Age', 'Experience', 'JobRole', 'Department'])

        # Predict salary
        predicted_salary = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f'Estimated Salary for {name}: ₹{predicted_salary:,.2f}')
    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
