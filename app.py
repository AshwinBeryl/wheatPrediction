from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('wheat.html')
    else:
        data=CustomData(
            area=request.form.get("area"),
            perimeter=request.form.get("perimeter"),
            compactness=request.form.get("compactness"),
            lengthOfKernel=request.form.get("lengthOfKernel"),
            widthOfKernel=request.form.get("widthOfKernel"),
            asymmetryCoefficent=request.form.get("asymmetryCoefficent"),
            lengthOfKernelGroove=request.form.get("lengthOfKernelGroove")
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('wheat.html',results=int(results[0]))
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        


