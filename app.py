from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
try:
    model_path = os.path.join(os.path.dirname(__file__), "model/loan_model.pkl")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("🔄 Prediction request received!")
    
    # Debug: Print all form data
    print("📝 Form data:")
    for key, value in request.form.items():
        print(f"  {key}: {value}")
    
    try:
        # Collect form data with validation
        no_of_dependents = int(request.form.get("no_of_dependents", 0))
        education = request.form.get("education", "")
        self_employed = request.form.get("self_employed", "")
        income_annum = float(request.form.get("income_annum", 0))
        loan_amount = float(request.form.get("loan_amount", 0))
        loan_term = int(request.form.get("loan_term", 0))
        cibil_score = float(request.form.get("cibil_score", 0))
        residential_assets_value = float(request.form.get("residential_assets_value", 0))
        commercial_assets_value = float(request.form.get("commercial_assets_value", 0))
        luxury_assets_value = float(request.form.get("luxury_assets_value", 0))
        bank_asset_value = float(request.form.get("bank_asset_value", 0))

        print(f"✅ Form data parsed successfully!")

        # Check if model is loaded
        if model is None:
            result = "❌ Model not loaded. Please check the model file."
            print(f"🔴 {result}")
            return render_template("index.html", prediction=result)

        # Create DataFrame with the same columns as training
        input_data = pd.DataFrame([{
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }])

        print(f"📊 Input data created: {input_data.iloc[0].to_dict()}")

        # Make prediction
        prediction = model.predict(input_data)[0]
        print(f"🔮 Raw prediction: {prediction}")
        
        # Get prediction probability for confidence
        try:
            prediction_proba = model.predict_proba(input_data)[0]
            confidence = max(prediction_proba) * 100
            print(f"📈 Confidence: {confidence:.1f}%")
        except Exception as prob_error:
            print(f"⚠️ Could not get probability: {prob_error}")
            confidence = None

        # Format result message (adjust based on your model's output)
        if prediction == 0:  # Adjust this based on your model
            if confidence:
                result = f"🎉 Loan Will be Approved!"
            else:
                result = "🎉 Loan Will be Approved!"
        else:
            if confidence:
                result = f"❌ Loan Might Get Rejected"
            else:
                result = "❌ Loan Might Get Rejected"

        print(f"✅ Final result: {result}")
        return render_template("index.html", prediction=result)

    except ValueError as ve:
        error_message = f"⚠️ Invalid input data: {str(ve)}"
        print(f"🔴 ValueError: {error_message}")
        return render_template("index.html", prediction=error_message)
    
    except Exception as e:
        error_message = f"⚠️ Error processing request: {str(e)}"
        print(f"🔴 General Error: {error_message}")
        return render_template("index.html", prediction=error_message)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)