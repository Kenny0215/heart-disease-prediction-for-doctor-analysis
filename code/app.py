from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/tool')
def tool(): return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])

        if age < 0 or age > 120:
            return render_template('predict.html', 
                                   prediction_text="Invalid Age Provided", 
                                   prob="0%", 
                                   prob_val=0, 
                                   res_color="secondary")
        
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        # 2. Scale and Predict
        final_features = np.array([features])
        scaled_features = scaler.transform(final_features)
        
        prediction = model.predict(scaled_features)
        
        # Calculate Probability for the progress bar
        # We use the probability of Class 1 (Disease)
        prob_raw = model.predict_proba(scaled_features)[0][1]
        prob_percent = prob_raw * 100

        if prediction[0] == 1:
            result = "Risk of Heart Disease Detected"
            color = "danger"
        else:
            result = "No Significant Risk Detected"
            color = "success"

        return render_template('predict.html', 
                               prediction_text=result, 
                               prob=f"{prob_percent:.2f}%", 
                               prob_val=int(prob_percent),
                               res_color=color)

    except Exception as e:
        return f"Form Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)