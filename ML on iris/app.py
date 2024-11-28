from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load(open('iris_model','rb'))

@app.route("/",methods=['GET'])
def Home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():


    float_features = [float(x) for x in request.form.values()]
        
    features = [np.array(float_features)]
    
    prediction = model.predict(features)

    if prediction[0] == 0:
        species = 'setosa'
    elif prediction[0] == 1:
        species = 'versicolor'
    else:
        species = 'virginica'

    return  render_template("pred.html",prediction_text = f"{species}")


if __name__ == "__main__":
    app.run(debug=True)