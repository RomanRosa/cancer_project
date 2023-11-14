import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


def save_object(file_path,obj):
    try:
        logging.info(f'Saving the object to : {file_path}')
        
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
        
        logging.info('Object Successfully Saved')

    except Exception as e:
        logging.info('Exception Occured While Saving Object In Utils.py File')
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        logging.info(f'Loading File Object From : {file_path}')
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured In Load Object From Utils.py')
        raise CustomException(e,sys)