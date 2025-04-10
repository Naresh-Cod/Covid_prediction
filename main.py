from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model and encoders
with open("model.pkl", "rb") as f:
    model, le_gender, le_cough, le_city, le_target, imputer = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = le_gender.transform([request.form['gender']])[0]
        fever = float(request.form['fever'])
        cough = le_cough.transform([request.form['cough']])[0]
        city = le_city.transform([request.form['city']])[0]
        
        fever = imputer.transform([[fever]])[0][0]
        input_data = np.array([[age, gender, fever, cough, city]])
        
        pred = model.predict(input_data)[0]
        prediction = le_target.inverse_transform([pred])[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
