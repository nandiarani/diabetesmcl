from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app= Flask(__name__)
Swagger(app)
CORS(app)

@app.route('/input/task',methods=['POST'])
def predict():
    """
    ini adalah endpoint
    ---
    tags:
        - Rest Controller
    parameters:
        -   name : body
            in: body
            required: true
            schema:
                id: diabetes
                required:
                    - Pregnancies
                    - Glucose
                    - BloodPressure
                    - SkinThickness
                    - Insulin
                    - BMI
                    - DiabetesPredigreeFunction
                    - Age
                properties:
                    Pregnancies:
                        type: int
                        description: Inputkan bilangan minimal 0
                        default: 0
                    Glucose:
                        type: int
                        description: Inputkan bilangan minimal 0
                        default: 0
                    BloodPressure:
                        type: int
                        description: Inputkan bilangan minimal 0
                        default: 0
                    SkinThickness:
                        type: int
                        description: Inputkan bilangan minimal 0
                        default: 0
                    Insulin:
                        type: int
                        description: Inputkan bilangan minimal 0
                        default: 0
                    BMI:
                        type: float
                        description: Inputkan bilangan minimal 0
                        default: 0
                    DiabetesPredigreeFunction:
                        type: float
                        description: Inputkan bilangan minimal 0
                        default: 0
                    Age:
                        type: int
                        description: Inputkan bilangan minimal 0
                        default: 0
    responses:
        200:
            description: Success input
    """
    Pregnancies= request.args['Pregnancies']
    Glucose= request.args['Glucose']
    BloodPressure= request.args['BloodPressure']
    SkinThickness= request.args['SkinThickness']
    Insulin= request.args['Insulin']
    BMI= request.args['BMI']
    DiabetesPredigreeFunction= request.args['DiabetesPredigreeFunction']
    Age= request.args['Age']

    X_New= np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPredigreeFunction,Age]])
    clf=joblib.load('Diabetes.pkl')
    resultPredict=clf[0].predict(X_New)
    return jsonify({'result': str(resultPredict[0])})

app.run(debug=True)
