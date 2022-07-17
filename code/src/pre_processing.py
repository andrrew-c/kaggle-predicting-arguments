
import os
import pandas as pd

from constants.constants import train_data_fpath

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.tree import DecisionTreeClassifier

# Scoring
from sklearn.metrics import roc_curve


def get_filenames(folder):

    pass



def get_train_data(fpath):

    """ Return dataframe of training data"""

    df = pd.read_csv(fpath)
    return df

def get_y_labels(y, target = 'discourse_effectiveness'):

    """ Using LabelEncoder"""

    # Create instance of label encoder
    le = LabelEncoder()

    # Fit label encoder using data
    y_ = le.fit_transform(y)

    # Return label encoder
    return y_



if __name__ == "__main__":

    # Get training data
    df = get_train_data(f"{train_data_fpath}")

    # Label encoder for Y
    le_target = LabelEncoder()

    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(df['discourse_text'], df['discourse_effectiveness'])

    # Encode targets as numbers
    y_train_ = le_target.fit_transform(y_train)
    y_test_ = le_target.transform(y_test)
    

    # Data pipeline
    pipe = Pipeline([('tfidf', TfidfVectorizer())
                    , ('pca', TruncatedSVD(n_components=5))  
                    , ('dtc', DecisionTreeClassifier())
                       ])

    pipe.fit(X_train, y_train_)

    # Score on training data
    print(f"Training data score = {pipe.score(X_train, y_train_):.2%}")

    # Score on test data
    print(f"Testing data score = {pipe.score(X_test, y_test_):.2%}")

    

