import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport
from IPython.display import display, HTML

from enum import Enum
import random

import Utils as U
import importlib
importlib.reload(U)

FILENAME_D = "./data/diabetes.csv"
FILENAME_A = "./data/alzheimer.csv"

SHUFFLE_SEED = 153
COLOR_SEED = 666
PD_MAX_ROWS = 10
PD_MAX_WIDTH = 400

TRAINING_CUT = 0.7 # 70%
SCIKIT_LEARN_RANDOM_STATE = 73

pd.set_option('display.max_rows', PD_MAX_ROWS)
pd.set_option('display.width', PD_MAX_WIDTH)

pd.set_option('future.no_silent_downcasting', True)

def print_column_types(df):
    print("Column types:")
    for column in df.columns:
        col_name = column.ljust(30)
        col_type = df[column].dtype
        print(f"{col_name} {col_type}")

def print_scores(y_true, y_pred):
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Presition
    precision = precision_score(y_true, y_pred)
    print(f"Precision: {precision:.2f}")
    
    # Recall
    recall = recall_score(y_true, y_pred)
    print(f"Recall: {recall:.2f}")

    # Calcular F1 Score
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.2f}")

    print()



class Custom_Dataframe:
    def __init__(self, name, df, tag_col_name, features_col_names):
        # Constructor de la clase
        self.name = name
        self.df = df
        self.tag_col_name = tag_col_name
        self.features_col_names = features_col_names

    def print_unique_tags(self):
        tags = self.df[self.tag_col_name].unique()
        print(f"Tags: {tags}")

    def replace_col_values(self, replace_dict, col_name):
        df = self.df
        df[col_name] = df[col_name].replace(replace_dict)

    def reduce_dimesionality(self, columns_to_keep):
        # U.printt("self.features_col_names", self.features_col_names)
        # U.printt("columns_to_keep", columns_to_keep)
        self.df = self.df[columns_to_keep]
        columns_to_keep.remove(self.tag_col_name)
        self.features_col_names = columns_to_keep

    def show(self):
        print(self.name)
        print(self.df)

        # Check types
        print_column_types(self.df)

        print()

        # Check unique tags
        self.print_unique_tags()        
        print()
        print()

    def print_tags_balance(self):
        print(self.name)
        total_rows = self.df.shape[0]
        print(f"Total rows = {total_rows}")
        tags = self.df[self.tag_col_name].unique()
        for tag in tags:
            count = self.df[self.df[self.tag_col_name] == tag].shape[0]
            percentage = 100 * count / total_rows
            formatted_percentage = "{:.2f}".format(percentage)
            print(f"Tag: {tag} -> {count} Rows ({formatted_percentage}%)")


class CYBERPUNK_COLORS(Enum):
    Red             = "#e74150"
    Yellow          = "#fee801"
    Green           = "#00ff9f"
    Dark_Blue       = "#005678"
    Night           = "#01012b"
    Cyan            = "#00ffe3"
    Red_Fuchsia     = "#ff1e61"
    Pink_Fuchsia    = "#ff008d"

    @classmethod
    def get_random_color(self, seed):
        random.seed(seed)        
        color_list = list(CYBERPUNK_COLORS)        
        selected_color = random.choice(color_list)
        return selected_color.value

class SEABORN_COLORMAPS(Enum):
    Viridis         = "viridis"
    Plasma          = "plasma"
    Inferno         = "inferno"
    Magma           = "magma"
    Cividis         = "cividis"
    Blues           = "Blues"
    Greens          = "Greens"
    Reds            = "Reds"
    Coolwarm        = "coolwarm"
    Spectral        = "Spectral"
    RdYlBu          = "RdYlBu"
    RdBu            = "RdBu"
    PiYG            = "PiYG"
    PRGn            = "PRGn"
    BrBG            = "BrBG"

    @classmethod
    def get_random_colormap(cls, seed=None):
        if seed is not None:
            random.seed(seed)        
        colormap_list = list(SEABORN_COLORMAPS)        
        selected_colormap = random.choice(colormap_list)
        return selected_colormap.value
    

# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# https://www.kaggle.com/datasets/brsdincer/alzheimer-features

def gen_cdf(filename, tag_col_name):
    filepath = Path(filename)

    # Checks read and the file
    if filepath.is_file():
        # Read .csv
        df = pd.read_csv(filepath)
        # Get non-tag columns
        features_col_names = [col for col in df.columns if col != tag_col_name]

        # Dataframe is shuffled to ease bias
        df = df.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

        # Create Dataframe class instance
        cdf = Custom_Dataframe(filename, df, tag_col_name, features_col_names)
        cdf.show()

        return cdf
    else:
        print(f"Need the file named: {filename}")

cdfd = gen_cdf(FILENAME_D, "Outcome")
cdfa = gen_cdf(FILENAME_A, "Group")


cdfa.df = cdfa.df[cdfa.df['Group'] != 'Converted']
cdfa.show()


cdfa.replace_col_values({'Nondemented': 0, 'Demented': 1}, cdfa.tag_col_name)
cdfa.replace_col_values({'F': 0, 'M': 1}, "M/F")
cdfa.show()

cdfd.print_tags_balance()
print()
cdfa.print_tags_balance()


columns_to_keep = ['Glucose', 'BMI', 'Outcome']
cdfd.reduce_dimesionality(columns_to_keep)

columns_to_keep = ['M/F', 'CDR', 'Group']
cdfa.reduce_dimesionality(columns_to_keep)

cdfd.show()
cdfa.show()


def tvt_df_split(df):
    train_df, aux_df = train_test_split(df, test_size=0.3, random_state=SCIKIT_LEARN_RANDOM_STATE)
    validation_df, test_df = train_test_split(aux_df, test_size=0.5, random_state=SCIKIT_LEARN_RANDOM_STATE)

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    return train_df, validation_df, test_df

# Diabetes custom dataframes
train_dfd, validation_dfd, test_dfd = tvt_df_split(cdfd.df)

train_cdfd = Custom_Dataframe(name="Diabetes Training Custom Dataframe", df=train_dfd, tag_col_name=cdfd.tag_col_name, features_col_names=cdfd.features_col_names)
validation_cdfd = Custom_Dataframe(name="Diabetes Validation Custom Dataframe", df=validation_dfd, tag_col_name=cdfd.tag_col_name, features_col_names=cdfd.features_col_names)
test_cdfd = Custom_Dataframe(name="Diabetes Testing Custom Dataframe", df=test_dfd, tag_col_name=cdfd.tag_col_name, features_col_names=cdfd.features_col_names)

train_cdfd.show()
validation_cdfd.show()
test_cdfd.show()


# Alzheimer custom dataframes
train_dfa, validation_dfa, test_dfa = tvt_df_split(cdfa.df)

train_cdfa = Custom_Dataframe(name="Alzheimer Training Custom Dataframe", df=train_dfa, tag_col_name=cdfa.tag_col_name, features_col_names=cdfa.features_col_names)
validation_cdfa = Custom_Dataframe(name="Alzheimer Validation Custom Dataframe", df=validation_dfa, tag_col_name=cdfa.tag_col_name, features_col_names=cdfa.features_col_names)
test_cdfa = Custom_Dataframe(name="Alzheimer Testing Custom Dataframe", df=test_dfa, tag_col_name=cdfa.tag_col_name, features_col_names=cdfa.features_col_names)

train_cdfa.show()
validation_cdfa.show()
test_cdfa.show()