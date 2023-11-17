import os
import sys
from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from src.utils import save_object, evaluate_model
from dataclasses import dataclass


@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

# ---------- 1 ---------------

    # Initiate Model Training Function
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent Features From Training and Testing Array')

            xtrain, ytrain, xtest, ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            # Instantiate Models
            models = {'LogisticRegression':LogisticRegression(),
                      'DecisionTree':DecisionTreeClassifier(),
                      'RandomForest':RandomForestClassifier(),
                      'GradientBoost':GradientBoostingClassifier(),
                      'KNN':KNeighborsClassifier(),
                      'SVM':SVC()}
            
            # Evaluate Models
            model_report,eval_df = evaluate_model(xtrain,ytrain,xtest,ytest,models)

            # Logging Report
            logging.info(f'Model Report : {model_report}')
            logging.info(f'Model Evaluation : \n{eval_df.to_string()}')

            # Best f1 Score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            # Best Model
            best_model = models[best_model_name]

            print('\n================================================================================================\n')

            print(f'Best Model Found : {best_model_name}, F1 Score In Testing : {best_model_score}')
            
            print
            ('\n================================================================================================\n')

            logging.info(f'Best Model Found : {best_model_name}, F1 Score In Testing : {best_model_score}')

            # Save Best Model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        # Add CustomException.py
        except Exception as e:
            logging.info('Exception Occured During Model Training')
            raise CustomException(e,sys)
