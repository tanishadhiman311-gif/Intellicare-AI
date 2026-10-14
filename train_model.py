
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Small educational example dataset.
# These labels are invented for software demonstration, not medical diagnosis.
data = {
    "symptoms": [
        "fever body ache fatigue",
        "fever headache body ache",
        "fever fatigue headache",
        "cough sore throat",
        "cough sore throat fatigue",
        "cough throat irritation",
        "headache nausea fatigue",
        "headache nausea",
        "nausea fatigue headache",
        "sneezing runny nose",
        "sneezing runny nose cough",
        "runny nose throat irritation",
        "fatigue body ache",
        "headache fatigue",
        "cough fatigue",
        "sneezing sore throat",
    ],
    "label": [
        "Pattern A", "Pattern A", "Pattern A",
        "Pattern B", "Pattern B", "Pattern B",
        "Pattern C", "Pattern C", "Pattern C",
        "Pattern D", "Pattern D", "Pattern D",
        "Pattern A", "Pattern C", "Pattern B", "Pattern D",
    ],
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["symptoms"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"],
)

model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("IntelliCare educational ML model trained!")
print(f"Test accuracy on tiny demo dataset: {accuracy:.2%}")
print("\nTest classification report:")
print(classification_report(y_test, predictions, zero_division=0))

with open("symptom_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nSaved model to symptom_model.pkl")
print("Important: this demo model is NOT medically validated.")
