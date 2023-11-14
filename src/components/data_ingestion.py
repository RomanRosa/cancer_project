import os
import sys
from src.logger import logging
from src.exception import CustomException

import pymongo
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

# Initialize data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

# Creating Data Ingestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()