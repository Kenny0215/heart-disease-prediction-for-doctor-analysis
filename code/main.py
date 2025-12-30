# Import functions from your other files
import joblib
from data_loader import load_data
from preprocessing import preprocess_data
from train_model import train_and_evaluate
from visualize import plot_results
from report import save_text_report

def main():
    print("STARTING PROJECT HEARTGUARD...")
    
    # 1. Load & Preprocess
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # 2. Train
    results, models = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 3. Visualize & Report
    plot_results(results)
    best_model_name = save_text_report(results, X_train.shape[0], X_test.shape[0])
    
    # 4. SAVE MODELS FOR DEPLOYMENT
    print(f"\nSaving the best model ({best_model_name}) and scaler...")
    
    best_model_object = models[best_model_name]
    
    # Save the model and the scaler
    joblib.dump(best_model_object, 'heart_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("FILES SAVED: heart_model.pkl, scaler.pkl")
    print("PROJECT COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()

# 
# def main():
#     print("STARTING PROJECT HEARTGUARD...")
    
#     # 1. Load
#     df = load_data()
    
#     # 2. Preprocess
#     X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
#     train_size = X_train.shape[0]
#     test_size = X_test.shape[0]
    
#     # 3. Train
#     results, models = train_and_evaluate(X_train, X_test, y_train, y_test)
    
#     # 4. Visualize
#     plot_results(results)

#     # 5. Report (Now captures the best model name from the report logic)
#     best_model_name = save_text_report(results, train_size, test_size)
#     best_model_object = models[best_model_name]

#     print(f"Saving models to disk...")
    
#     # Save the best model
#     joblib.dump(best_model_object, 'heart_attack_model.pkl')
    
#     # Save the scaler (EXTREMELY IMPORTANT for future predictions)
#     joblib.dump(scaler, 'scaler.pkl')
    
#     print("Files saved: heart_attack_model.pkl, scaler.pkl")
    
#     # --- STEP 6: USER PREDICTION ---
#     print(f"\nThe best model determined is: {best_model_name}")
    
#     # Get the actual model object based on the name returned from report.py
#     best_model_object = models[best_model_name]
    
#     # Ask user if they want to test
#     while True:
#         choice = input(f"\nDo you want to enter patient data for prediction using {best_model_name}? (y/n): ").lower()
#         if choice == 'y':
#             get_user_data_and_predict(best_model_object, scaler)
#         elif choice == 'n':
#             break
#         else:
#             print("Invalid input.")

#     print("\nPROJECT COMPLETED SUCCESSFULLY.")


# if __name__ == "__main__":
#     main()
#