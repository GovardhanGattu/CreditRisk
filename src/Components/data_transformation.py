import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomeException
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif,SelectKBest
from imblearn.over_sampling import ADASYN
from src.utils import saveobject
from src.logger import logging






@dataclass
class DataTrasformationConfig:
    train_data_path=os.path.join('artifacts','traindata.csv')
    test_data_path=os.path.join('artifacts','testdata.csv')
    preprocessor_obj_filepath=os.path.join('artifacts','Preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.datatransformationconfig=DataTrasformationConfig()

    
    def get_transformation_object(self,data):
        try:
            columns = data.columns
            pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer([
                ('pipeline',pipeline,columns)
            ]
            )

            return preprocessor
        except Exception as e:
            raise CustomeException(e,sys)

    def initiate_data_transformation(self,data_path):
        try:
            train_df,test_df = self.initiate_feature_selection(data_path)
            target_column="credit_risk"
            input_features_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]
            input_features_test_df=test_df.drop(columns=[target_column],axis=1)
            target_column_test_df=test_df[target_column]
            preprocessor_obj=self.get_transformation_object(input_features_test_df)
            
            train_df_arr=preprocessor_obj.fit_transform(input_features_train_df)
            test_df_arr=preprocessor_obj.transform(input_features_test_df)

            train_arr=np.c_[train_df_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[test_df_arr,np.array(target_column_test_df)]

            saveobject(
                filepath=self.datatransformationconfig.preprocessor_obj_filepath,
                object=preprocessor_obj

            )
            logging.info(f"Preprocessor pickle file is saved successfully")
            return(
                train_arr,
                test_arr,
                self.datatransformationconfig.preprocessor_obj_filepath
            )

        except Exception as e:
            raise CustomeException(e,sys)
    
    def initiate_feature_selection(self,data_path):
        df=pd.read_csv(data_path)
        X_train,X_test,y_train,y_test = train_test_split(df.drop(labels=['credit_risk'],axis=1),df['credit_risk'],test_size=0.2,random_state=33)
        imp_features = SelectKBest(mutual_info_classif, k=10)
        imp_features.fit(X_train, y_train)
        features = X_train.columns[imp_features.get_support()]
        print(f"Features after the feature selection are{features}")
        imp_features_df=pd.DataFrame(df,columns=features)
        imp_features_df['credit_risk']=df['credit_risk']

        train_data,test_data = self.handle_imbalanced_data(imp_features_df)


        return(
            train_data,
            test_data
        )

    
    def handle_imbalanced_data(self,featured_data):
        X_train,X_test,y_train,y_test = train_test_split(featured_data.drop(labels=['credit_risk'],axis=1),featured_data['credit_risk'],test_size=0.2,random_state=33)
        adasyn = ADASYN(random_state=42)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
        train_df=pd.DataFrame(X_train_resampled,columns=featured_data.columns)
        train_df['credit_risk']=y_train_resampled
        test_df=pd.DataFrame(X_test,columns=featured_data.columns)
        test_df['credit_risk']=y_test

        train_df.to_csv(self.datatransformationconfig.train_data_path,index=False,header=True)
        test_df.to_csv(self.datatransformationconfig.test_data_path,index=False,header=True)

        return(
            train_df,
            test_df
        )

        
        


