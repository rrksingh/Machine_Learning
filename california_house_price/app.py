from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('california_house_price')

@app.route("/",methods=['GET'])
def Home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():


    float_features = [float(x) for x in request.form.values()]
        
    features = [np.array(float_features)]
    
    prediction = model.predict(features)

    predict=prediction[0]
    

    return  render_template("index.html",prediction_text = f"{predict}")


if __name__ == "__main__":
    app.run(debug=True)