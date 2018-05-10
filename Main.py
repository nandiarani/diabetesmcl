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
    new_task= request.get_json()
    Pregnancies= new_task['Pregnancies']
    Glucose= new_task['Glucose']
    BloodPressure= new_task['BloodPressure']
    SkinThickness= new_task['SkinThickness']
    Insulin= new_task['Insulin']
    BMI= new_task['BMI']
    DiabetesPredigreeFunction= new_task['DiabetesPredigreeFunction']
    Age= new_task['Age']

    X_New= np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPredigreeFunction,Age]])
    clf=joblib.load('Diabetes.pkl')
    resultPredict=clf[0].predict(X_New)
    return jsonify({'message': str(resultPredict)})


