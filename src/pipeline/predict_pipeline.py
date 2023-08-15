import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path= os.path.join("artifacts","model.pkl")
            preprocessor_path= os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        city: str,
        room_type:str,
        person_capacity: int,
        multiple_rooms: str,
        business: str,
        cleanliness_rating: int,
        guest_satisfaction: int,
        bedrooms: int,
        city_center_km: float,
        metro_distance_km: float):



        self.city = city
        self.room_type =  room_type
        self.person_capacity = person_capacity
        self.multiple_rooms = multiple_rooms 
        self.business = business
        self.cleanliness_rating = cleanliness_rating 
        self.guest_satisfaction = guest_satisfaction
        self.bedrooms = bedrooms
        self.city_center_km = city_center_km
        self.metro_distance_km = metro_distance_km

        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "City": [self.city],
                "Room Type": [self.room_type],
                "Person Capacity": [self.person_capacity],
                "Multiple Rooms": [self.multiple_rooms],
                "Business": [self.business],
                "Cleanliness Rating": [self.cleanliness_rating],
                "Guest Satisfaction": [self.guest_satisfaction],
                "Bedrooms": [self.bedrooms],
                "City Center (km)": [self.city_center_km],
                "Metro Distance (km)": [self.metro_distance_km]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)