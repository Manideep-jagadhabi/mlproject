import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer #to handle missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") # path for input to data tranformation component


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):   # to create pikle file which is responsible for catogorical to numerical transformation
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(           #creating a numerical pipeline also to handle missingvalues
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #this handles missing values
                ("scaler",StandardScaler())  #

                ]
            )

            cat_pipeline=Pipeline(       #creating a Catagorical pipeline also to handle missingvalues

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),  #this handles missing values by replacing with most frequent values
                ("one_hot_encoder",OneHotEncoder()), #encoding  Converts categorical variables into a format that can be used in ML models.
                ("scaler",StandardScaler(with_mean=False)) #Standardizes numerical features so that they have a similar range, improving model performance.
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}") #numerical columns standard scaling completed
            logging.info(f"Numerical columns: {numerical_columns}")  #categorical columns encoding completed

            preprocessor=ColumnTransformer(  #here combining numerical and categorical pipelines with this function
                [
                ("num_pipeline",num_pipeline,numerical_columns), #name,pipeline,columns
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor         #returned the data after Cat and Num transformation
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)  #read train and test data from data ingestion
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")  

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() #creating obj for cat and num Transformation

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)  #calling pkl and doing fit transform on training data and transform on test data
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[           #   This code concatenates input features and target labels column-wise using `np.c_[]`, creating `train_arr` and `test_arr`, where the last column represents the target variable. It ensures the data is structured properly for model training and evaluation.
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
           #Now, train_arr and test_arr are ready for machine learning models where the last column represents the target labels.
            logging.info(f"Saved preprocessing object.")

            save_object(  #from utils saves everything

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path, #pklfile
            )
        except Exception as e:
            raise CustomException(e,sys)