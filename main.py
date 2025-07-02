import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

df=pd.read_csv("PhishingEmailData.csv", encoding='ISO-8859-1')
print(df.columns)
print(df.head(2))

# Filter only Email_Content and To
df = df[['Email_Content', 'To']].dropna()

# Define label: 1 for phishing, 0 for legit
df['Label'] = df['To'].apply(lambda x: 1 if x.strip() == 'NB' else 0)

# Features and target
X = df['Email_Content']
y = df['Label']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict from user input
email = "Please update your payment info to avoid account suspension."
email_vec = vectorizer.transform([email])
score = model.decision_function(email_vec)[0]  # Get raw score
pred = model.predict(email_vec)[0]

confidence = abs(score) / 10
confidence = min(round(confidence * 100, 2), 100)

print("PHISHING" if pred == 1 else "Legit")
print(f"Confidence: {confidence}%")

import pickle

with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
