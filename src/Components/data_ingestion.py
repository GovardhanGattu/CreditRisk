import sys
import os
import pandas as pd
from src.exception import CustomeException
from dataclasses import dataclass
from src.Components.data_transformation import DataTransformation,DataTrasformationConfig
from src.Pipeline.model_trainer_pipeline import ModelTrainer,ModelTrainerConfig




@dataclass
class DataIngestionConfig():
    actual_data_path=os.path.join('artifacts','GermanData.csv')

class DataIngestion():
    def __init__(self):
        self.dataingestionconfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            os.makedirs(os.path.dirname(self.dataingestionconfig.actual_data_path),exist_ok=True)
            df=pd.read_csv("Data/GermanData.csv")
            df.to_csv(self.dataingestionconfig.actual_data_path,index=False,header=True)

            return(
                self.dataingestionconfig.actual_data_path
            )
        except Exception as e:
            raise CustomeException(e,sys)
        
if __name__=="__main__":
    dataobj =DataIngestion()
    dataset_path=dataobj.initiate_data_ingestion()

    datatransformation=DataTransformation()
    train_arr,test_arr,_=datatransformation.initiate_data_transformation(dataset_path)

    model =ModelTrainer()
    model.initiate_model_training(train_arr,test_arr)
    
