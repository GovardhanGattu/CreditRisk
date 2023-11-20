from flask import Flask,request,render_template
import pandas as pd
from src.exception import CustomeException
from src.logger import logging
from src.Pipeline.predict_pipeline import PredictPipeline,CustomData

application =Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predit_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            status=request.form.get('status'),
            duration=request.form.get('duration'),
            credit_history=request.form.get('credit_history'),
            purpose=request.form.get('purpose'),
            amount=request.form.get('amount'),
            savings=request.form.get('savings'),
            employment_duration=request.form.get('employment_duration'),
            property=request.form.get('property'),
            age=request.form.get('age'),
            other_installment_plans=request.form.get('other_installment_plans')
        )


        dataframe=data.prepare_data_frame()
        predict_userdata=PredictPipeline()
        result=predict_userdata.predict_pipeline(dataframe)

        if result==0:
            prediction= 'Bad'
        elif result==1:
            prediction ='Good'

        return render_template('home.html',result=f"The Credit risk for the entered data is {prediction}")
    

if __name__=="__main__":
    app.run(host="0.0.0.0")      