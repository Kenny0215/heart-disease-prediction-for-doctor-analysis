from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n--- Step 3: Training Models with Tuning ---")
    print("(This might take a few seconds...)\n")
    
    # 1. Logistic Regression (Standard)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_acc = accuracy_score(y_test, log_reg.predict(X_test))
    
    # 2. Decision Tree (Standard)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    
    # 3. Random Forest (Advanced Tuning)
    # We define a "Grid" of settings to test
    param_grid = {
        'n_estimators': [50, 100, 200],      # Try different numbers of trees
        'max_depth': [None, 10, 20, 30],     # Try different depths
        'min_samples_split': [2, 5, 10],     # Try different split rules
        'random_state': [42]
    }
    
    rf = RandomForestClassifier()
    # GridSearchCV tests all combinations to find the best one
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    # Get the best version of Random Forest found
    best_rf = grid_search.best_estimator_
    rf_acc = accuracy_score(y_test, best_rf.predict(X_test))
    
    print(f"Logistic Regression Accuracy : {log_acc:.4f}")
    print(f"Decision Tree Accuracy       : {dt_acc:.4f}")
    print(f"Random Forest (Tuned) Accuracy: {rf_acc:.4f}")
    print(f"  -> Best Settings Found: {grid_search.best_params_}")

    # Store models and results
    models = {
        "Logistic Regression": log_reg,
        "Decision Tree": dt,
        "Random Forest": best_rf 
    }
    
    results = {
        "Logistic Regression": log_acc,
        "Decision Tree": dt_acc,
        "Random Forest": rf_acc
    }

    return results, models