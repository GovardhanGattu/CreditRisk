import sys
import os
import pandas as pd
from src.exception import CustomeException
from src.logger import logging
from src.utils import loadobject





class PredictPipeline():
    def __init__(self):
        pass

    def predict_pipeline(self,inputdata):
        try:
            print(inputdata)
            model_path=os.path.join('artifacts','model.pkl')
            model=loadobject(filepath=model_path)
            preprocessor_path=os.path.join('artifacts','Preprocessor.pkl')
            preprocessor=loadobject(filepath=preprocessor_path)
            scaled_data=preprocessor.transform(inputdata)
            model_prediction=model.predict(scaled_data)
            
            return model_prediction
        
            
        except Exception as e:
            raise CustomeException(e,sys)


class CustomData():
    def __init__(self,
                 status: int,
                 duration:int,
                 credit_history:int,
                 purpose:int,
                 amount:int,
                 savings:int,
                 employment_duration: int,
                 property:int,
                 age:int,
                 other_installment_plans:int,
                 ):
        
        self.status=status
        self.duration=duration
        self.credit_history=credit_history
        self.purpose=purpose
        self.amount=amount
        self.savings=savings
        self.employment_duration=employment_duration
        self.property=property
        self.age=age
        self.other_installment_plans=other_installment_plans
        
        

    def prepare_data_frame(self):
        try:
            custom_data={
                'status':[self.status],
                'duration':[self.duration],
                'credit_history':[self.credit_history],
                'purpose':[self.purpose],
                'amount': [self.amount],
                'savings':[self.savings],
                'employment_duration':[self.employment_duration],
                'property':[self.property],
                'age': [self.age],
                'other_installment_plans':[self.other_installment_plans],
            }

            custom_dataframe=pd.DataFrame(custom_data)

            return custom_dataframe
        
        except Exception as e:
            raise CustomeException(e,sys)
