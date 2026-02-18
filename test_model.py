import unittest
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from model_logic import fit_model, predict_model

class TestDeepLearningModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Trainings- und Testdaten aus Dateien."""
        train_df = pd.read_csv('train_data.csv')
        test_df = pd.read_csv('test_data.csv')

        cls.X_train = train_df.drop('Class', axis=1)
        cls.y_train = train_df['Class']
        cls.X_test = test_df.drop('Class', axis=1)
        cls.y_test = test_df['Class']
        cls.norm_fit_time = 0.5

    def test_1_predict_accuracy(self):
        """
        Testfall 1: Test der Vorhersagefunktion (predict).
        Indikator: Accuracy auf den Testdaten aus test_data.csv.
        Ziel: Accuracy > 0.95.
        """
        model, scaler = fit_model(self.X_train, self.y_train)
        predictions = predict_model(model, scaler, self.X_test)

        acc = accuracy_score(self.y_test, predictions)

        print(f"\n[Test predict()] Gemessene Accuracy: {acc:.4f}")
        self.assertGreater(acc, 0.95, f"Accuracy zu niedrig: {acc} < 0.95")

    def test_2_fit_runtime(self):
        """
        Testfall 2: Überprüfung der Laufzeit der Trainingsfunktion (fit).
        Ziel: Laufzeit < 120% der repräsentativen Normzeit.
        """
        start_time = time.time()
        fit_model(self.X_train, self.y_train)
        duration = time.time() - start_time

        limit = self.norm_fit_time * 1.2 # 120% der Normzeit
        print(f"\n[Test fit()] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")

        self.assertLess(duration, limit, f"Training dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
