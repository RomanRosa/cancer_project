import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass

# Sklearn Pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:    
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()