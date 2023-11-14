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