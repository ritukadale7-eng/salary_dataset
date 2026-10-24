import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained linear regression model
model = pickle.load(open("linear.pkl", "rb"))

# Embedded HTML/CSS UI Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Salary Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #333;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 420px;
            text-align: center;
            backdrop-filter: blur(10px);
        }

        h2 {
            color: #1a202c;
            font-weight: 700;
            margin-bottom: 8px;
            font-size: 26px;
        }

        p.subtitle {
            color: #718096;
            font-size: 14px;
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }

        label {
            display: block;
            font-weight: 500;
            font-size: 13px;
            color: #4a5568;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input[type="number"] {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s ease;
            outline: none;
        }

        input[type="number"]:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
        }

        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s ease, opacity 0.2s ease;
            margin-top: 10px;
        }

        button:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        button:active {
            transform: translateY(1px);
        }

        .result-card {
            margin-top: 25px;
            padding: 16px;
            background: #edf2f7;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            animation: fadeIn 0.4s ease-in-out;
        }

        .result-card h3 {
            color: #2d3748;
            font-size: 18px;
            font-weight: 600;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="container">
        <h2>Salary Predictor</h2>
        <p class="subtitle">Enter your experience to estimate expected compensation</p>
        
        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="experience">Years of Experience</label>
                <input type="number" step="0.1" id="experience" name="experience" placeholder="e.g. 3.5" value="{{ user_input if user_input is defined else '' }}" required>
            </div>
            <button type="submit">Predict Salary</button>
        </form>

        {% if prediction_text %}
        <div class="result-card">
            <h3>{{ prediction_text }}</h3>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""


@app.route("/")
def home():
  return render_template_string(HTML_TEMPLATE, prediction_text=None)


@app.route("/predict", methods=["POST"])
def predict():
  try:
    # Get years of experience from form input
    experience = float(request.form["experience"])

    # Model was trained with 'Unnamed: 0' and 'YearsExperience' columns
    input_df = pd.DataFrame(
        [[0.0, experience]], columns=["Unnamed: 0", "YearsExperience"]
    )

    # Make prediction
    prediction = model.predict(input_df)[0]

    return render_template_string(
        HTML_TEMPLATE,
        prediction_text=f"Estimated Salary: ${prediction:,.2f}",
        user_input=experience,
    )

  except Exception as e:
    return render_template_string(
        HTML_TEMPLATE, prediction_text=f"Error in prediction: {str(e)}"
    )


if __name__ == "__main__":
  app.run(debug=True)
