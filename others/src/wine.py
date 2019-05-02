import os
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def train(train_dir: Path, model_dir: Path):
    df = pd.read_csv(train_dir/"wine.csv", engine='python')

    X = df.drop('y', axis=1)
    y = df['y']

    clf = RandomForestClassifier()
    clf.fit(X, y)

    joblib.dump(clf,  model_dir/"model.joblib")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()

    train_dir = Path(args.train)
    model_dir = Path(args.model_dir)
    train(train_dir, model_dir)


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    model_path = Path(model_dir)/"model.joblib"
    clf = joblib.load(model_path)
    return clf