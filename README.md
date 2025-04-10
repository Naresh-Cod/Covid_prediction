# 🧠 COVID Prediction Flask App

This project is a simple machine learning-powered web application built with **Flask**. It predicts whether a person is likely to have COVID based on symptoms like fever, cough type, age, gender, and location.

---

## 📦 Features

- Predicts COVID status using a trained Random Forest model.
- Takes user input through a web form.
- Handles missing values automatically.
- Encodes categorical data using `LabelEncoder`.

---

## 📁 Project Structure

covid_predictor/ │ ├── model.py # Train and save ML model to model.pkl ├── main.py # Flask app to serve prediction ├── model.pkl # Trained model ├── requirements.txt # Python dependencies ├── templates/ │ └── index.html # Web form └── .github/workflows/ └── python-app.yml # GitHub CI/CD# Covid_prediction
