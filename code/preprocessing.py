from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    print("\n--- Step 2: Preprocessing Data ---")
    
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    # Fit on training data
    X_train = scaler.fit_transform(X_train)
    # Transform test data
    X_test = scaler.transform(X_test)

    print("Data split and scaled.")
    
    # RETURN THE SCALER HERE
    return X_train, X_test, y_train, y_test, scaler