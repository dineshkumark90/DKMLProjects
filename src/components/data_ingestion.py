import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


@dataclass
class dataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts', "data.csv")
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")

class dataIngestion:
    def __init__(self):
        self.ingestionConfig = dataIngestionConfig()

    def initiateDataIngestion(self):
        logging.info("Ingesting Data")
        try:
            df = pd.read_csv('notebook\dataspi.csv')
            logging.info("Read dataset as dataframe complete")

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)

            df.to_csv(self.ingestionConfig.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestionConfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestionConfig.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed") 

            return(
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj =  dataIngestion()
    train_data, test_data=obj.initiateDataIngestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)

    