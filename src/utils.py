import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException

def save_object(file_path, obj):
    """Saves a Python object to the specified file path using pickle."""
    try:
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)