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
        return render_template('modelpage.html')
    else:
        data=CustomData(

            city=request.form.get('city'),
            room_type=request.form.get('Room_Type'),
            person_capacity =int(request.form.get('person_capacity')),
            multiple_rooms = int(request.form.get('Multiple_Rooms')),
            business = int(request.form.get("bussiness")),
            cleanliness_rating = int(request.form.get("Cleanliness_Rating")),
            guest_satisfaction = int(request.form.get("Guest_Satisfaction")),
            bedrooms = int(request.form.get("Bedrooms")),
            city_center_km = float(request.form.get("City_Center_(km)")),
            metro_distance_km = float(request.form.get("Metro_Distance_(km)"))

        )


        pred_df=data.get_data_as_data_frame()
        print(pred_df.head())
        print("Before Prediction")


        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('modelpage.html',results=round(results[0],2))
    

if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host = "0.0.0.0" , port=port)        
