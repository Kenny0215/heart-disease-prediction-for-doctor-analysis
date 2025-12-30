import datetime

def save_text_report(results, train_size, test_size):
    print("\n--- Step 5: Generating Text Report ---")
    
    filename = "AI_Project_Result.txt"
    
    # Get current time for the report
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Find the best model with tie-breaking
    best_accuracy = max(results.values())
    
    # If there's a tie (multiple models with same accuracy), use a preference order
    tied_models = [model for model, acc in results.items() if acc == best_accuracy]
    
    if len(tied_models) > 1:
        # Define preference order for tie-breaking
        # Random Forest is first preference
        preference_order = ["Random Forest", "Decision Tree", "Logistic Regression"]
        
        # Find the first model in preference order that's in the tied models
        best_model = tied_models[0] # Default fallback
        for model in preference_order:
            if model in tied_models:
                best_model = model
                break
    else:
        best_model = tied_models[0]

    with open(filename, "w") as f:
        f.write("==================================================\n")
        f.write("      PROJECT HEARTGUARD - PERFORMANCE REPORT     \n")
        f.write("==================================================\n")
        f.write(f"Date Generated: {now}\n")
        f.write("Dataset: Heart Disease UCI\n")
        f.write("==================================================\n\n")

        f.write("DATASET INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Set Size : {train_size} samples (80%)\n")
        f.write(f"Testing Set Size  : {test_size} samples (20%)\n")
        f.write(f"Total Samples     : {train_size + test_size}\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        for model_name, acc in results.items():
            f.write(f"{model_name:<25} : {acc*100:.2f}%\n")
            
        f.write("-" * 40 + "\n\n")
        
        f.write("CONCLUSION:\n")
        f.write(f"The best performing model is: {best_model}\n")
        f.write(f"Accuracy achieved: {best_accuracy*100:.2f}%\n")
        
        if len(tied_models) > 1:
            f.write(f"Note: Multiple models achieved {best_accuracy*100:.2f}% accuracy.\n")
            f.write(f"Selected {best_model} based on preference order for production use.\n")
        
        f.write("\n==================================================\n")
        f.write("        End of Automated Report\n")
        f.write("==================================================\n")

    print(f"Report saved successfully as '{filename}'")
    
    # --- CHANGE: RETURN THE CHOSEN MODEL ---
    return best_model