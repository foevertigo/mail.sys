from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        email = request.form["email"]
        vec = vectorizer.transform([email])
        pred = model.predict(vec)[0]
        score = model.decision_function(vec)[0]
        confidence = min(round(abs(score) / 10 * 100, 2), 100)
        result = {
            "prediction": "PHISHING" if pred == 1 else "Legit",
            "confidence": confidence
        }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
