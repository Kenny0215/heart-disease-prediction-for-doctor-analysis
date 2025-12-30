import pandas as pd
import numpy as np

# --- HELPER FUNCTION TO HANDLE INPUT LOOPS ---
def get_valid_input(prompt):
    """
    This function keeps asking the user for input until they enter a valid number.
    It acts like a 'do-while' loop for every single question.
    """
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("[!] Invalid input. Please enter a number (e.g., 1, 0, 55.5). Try again.")

# --- MAIN PREDICTION FUNCTION ---
def get_user_data_and_predict(model, scaler):
    print("\n=============================================")
    print("      HEART DISEASE PREDICTION SYSTEM        ")
    print("=============================================")
    print("Please enter the following patient details:")
    print("(Enter numeric values only)\n")

    # We removed the big 'try-except' block because the 
    # helper function handles errors individually now.

    # 1. Age
    age = get_valid_input("1. Age (years): ")
    
    # 2. Sex
    print("\n--- Sex ---")
    print("1 = Male")
    print("0 = Female")
    sex = get_valid_input("2. Select Sex (0 or 1): ")
    
    # 3. Chest Pain
    print("\n--- Chest Pain Type ---")
    print("0 = Typical Angina (Pain during exercise/stress)")
    print("1 = Atypical Angina (Pain not matching classic signs)")
    print("2 = Non-anginal Pain (Likely not heart related)")
    print("3 = Asymptomatic (No pain)")
    cp = get_valid_input("3. Select Chest Pain Type (0-3): ")
    
    # 4. BP
    trestbps = get_valid_input("\n4. Resting Blood Pressure (mm Hg, e.g., 120): ")
    
    # 5. Chol
    chol = get_valid_input("5. Serum Cholestoral (mg/dl, e.g., 200): ")
    
    # 6. FBS
    print("\n--- Fasting Blood Sugar ---")
    print("1 = True (Sugar > 120 mg/dl)")
    print("0 = False (Sugar <= 120 mg/dl)")
    fbs = get_valid_input("6. Select Fasting Blood Sugar (0 or 1): ")
    
    # 7. ECG
    print("\n--- Resting ECG Results ---")
    print("0 = Normal")
    print("1 = ST-T wave abnormality")
    print("2 = Left ventricular hypertrophy")
    restecg = get_valid_input("7. Select Resting ECG results (0-2): ")
    
    # 8. Max Heart Rate
    thalach = get_valid_input("\n8. Maximum Heart Rate achieved (e.g., 150): ")
    
    # 9. Exercise Angina
    print("\n--- Exercise Induced Angina ---")
    print("1 = Yes (Pain caused by exercise)")
    print("0 = No")
    exang = get_valid_input("9. Select Exercise Induced Angina (0 or 1): ")
    
    # 10. Oldpeak
    oldpeak = get_valid_input("\n10. ST depression induced by exercise (e.g., 0.0 to 6.0): ")
    
    # 11. Slope
    print("\n--- Slope of the peak exercise ST segment ---")
    print("0 = Upsloping (Better/Healthy)")
    print("1 = Flat")
    print("2 = Downsloping (Worse)")
    slope = get_valid_input("11. Select Slope (0-2): ")
    
    # 12. CA (Vessels)
    print("\n--- Major Vessels (Fluoroscopy) ---")
    print("Count of major vessels (0-3) colored by fluoroscopy.")
    print("NOTE: Type '0' if you do not have a medical report.")
    print("0 = No calcification (Normal)")
    print("1-3 = Number of calcified vessels")
    ca = get_valid_input("12. Enter number of major vessels (0-4): ")
    
    # 13. Thalassemia
    print("\n--- Thalassemia ---")
    print("0 = Null/Dropped")
    print("1 = Fixed Defect")
    print("2 = Normal")
    print("3 = Reversable Defect")
    thal = get_valid_input("13. Select Thalassemia (0-3): ")

    # --- PROCESS DATA ---
    user_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])

    # Scale
    user_data_scaled = scaler.transform(user_data)

    # Predict
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)

    print("\n=============================================")
    print("             PREDICTION RESULT               ")
    print("=============================================")
    
    confidence = np.max(probability) * 100

    if prediction[0] == 1:
        print(f"RESULT: HIGH RISK of Heart Disease detected.")
        print(f"Model Confidence: {confidence:.2f}%")
        print("Please consult a cardiologist immediately.")
    else:
        print(f"RESULT: LOW RISK (Healthy).")
        print(f"Model Confidence: {confidence:.2f}%")
        print("Maintain a healthy lifestyle!")
    print("=============================================\n")