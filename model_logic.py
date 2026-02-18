import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging
import time
import os
from functools import wraps

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def my_logger(orig_func):
    """Loggt den Funktionsnamen und die übergebenen Argumente."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f'Ran with args: {args}, and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)
    return wrapper

def my_timer(orig_func):
    """Loggt die Ausführungszeit der Funktion."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        logging.info(f'{orig_func.__name__} ran in: {t2:.4f} sec')
        return result
    return wrapper

@my_logger
@my_timer
def load_data(filepath):
    """Lädt die Bank Note Daten."""
    try:
        data = pd.read_csv(filepath)
        X = data.drop('Class', axis=1)
        y = data['Class']
        return X, y
    except Exception as e:
        logging.error(f"Fehler beim Laden der Daten: {e}")
        raise

@my_logger
@my_timer
def fit_model(X_train, y_train):
    """Skaliert Daten und trainiert ein Neuronales Netz (MLP)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # MLPClassifier als Alternative zum alten DNNClassifier
    model = MLPClassifier(hidden_layer_sizes=(10, 20, 10), max_iter=500, random_state=101)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

@my_logger
@my_timer
def predict_model(model, scaler, X_test):
    """Führt Vorhersagen mit dem skalierten Modell durch."""
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    return predictions

if __name__ == "__main__":
    X, y = load_data('bank_note_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    model, scaler, duration = fit_model(X_train, y_train)
    preds = predict_model(model, scaler, X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
    logging.info(f"Deep Learning Accuracy im Testlauf: {acc}")
