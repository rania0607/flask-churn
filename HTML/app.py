from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Charger le modèle
model = pickle.load(open('C:/Users/dell/Desktop/python/Machine learning/HTML/rf.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extraire les données du formulaire
        credit_score = float(request.form['creditscore'])
        age = float(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        num_products = int(request.form['numofproducts'])
        has_cr_card = int(request.form['hascrcard'])
        is_active = int(request.form['isactivemember'])
        est_salary = float(request.form['estimatedsalary'])
        gender = request.form['gender']
        geography = request.form['geography']

        # Encoder Gender
        gender_male = 1 if gender == 'Male' else 0

        # Encoder Geography (on suppose que le modèle a été entraîné avec une seule dummy 'Geography_Germany')
        geography_germany = 1 if geography == 'Germany' else (0 if geography == 'France' else None)

        # Créer le vecteur d'entrée
        features = np.array([[credit_score, age, tenure, balance, num_products,
                              has_cr_card, is_active, est_salary,
                              gender_male, geography_germany]])

        # Prédire
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"Erreur: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
