from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Initialize Flask app
app = Flask(__name__,static_folder='static')

# Load data and preprocess (from your Jupyter notebook code)
crop = pd.read_csv("Crop_recommendation.csv")
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Map label to numeric
crop['crop_num'] = crop['label'].map(crop_dict)
crop.drop(['label'], axis=1, inplace=True)

# Prepare features and target
X = crop.drop(['crop_num'], axis=1)
y = crop['crop_num']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model using pickle (optional, but for persistence)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)


# Prediction function
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    transformed_features = scaler.transform(features)
    prediction = model.predict(transformed_features)
    return prediction[0]


# Flask Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Make a prediction
        prediction = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)

        # Mapping the prediction to crop name
        crop_name = list(crop_dict.keys())[list(crop_dict.values()).index(prediction)]

        # Construct image file name
        image_file = f"{crop_name}.jpg"  # Make sure this image exists in /static/

        result = f"The best crop to cultivate is: {crop_name.capitalize()}"
        return render_template('index.html', result=result, image_file=image_file)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
