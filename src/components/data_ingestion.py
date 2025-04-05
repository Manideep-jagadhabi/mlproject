import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd  # need to deal with Data Frames

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:  #to give input
    train_data_path: str=os.path.join('artifacts',"train.csv")  #giving path and all outputs saves inside artiacts folder and file name is train.csv
    test_data_path: str=os.path.join('artifacts',"test.csv")  # this 3 lines are the inputs we are giving in Data ingestion component
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()       #Creating a variable to store all of the inputs above

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')  #Reading Dataset
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)  #  #Creation of directeries form train,test,raw
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #Converted Rawdata to CSV 

            logging.info("Train test split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)  #Check google Keep day 4

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)  # Saved Train file
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) #saved Tesdt File

            
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,   #returning train and test data for transformation
                self.ingestion_config.test_data_path

            )

        except Exception as e:
            raise  CustomException(e,sys)
        
if __name__=="__main__":    # This ensures that DataIngestion() and initiate_data_ingestion() run only when the script is executed directly, not when imported into another script.
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion() 
  # combined data ingestion

  #combined data transformation
    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))  



