import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Параметры
DATA_PATH = 'cardio_train.csv'
OUTPUT_FILE = 'model.bin'
MAX_DEPTH = 4
LEARNING_RATE = 0.05
N_ESTIMATORS = 200
SEED = 1

def load_data(path):
    df = pd.read_csv(path, sep=';')
    df = df.drop(columns='id')
    df['age'] = (df['age'] / 365.25).round(1)
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

    df = df[
        df['ap_hi'].between(70, 250)
        & df['ap_lo'].between(40, 150)
        & (df['ap_hi'] > df['ap_lo'])
        & df['height'].between(120, 220)
        & df['weight'].between(30, 200)
        & df['bmi'].between(15, 60)
    ]
    return df


def train(df_train, y_train):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient='records'))

    model = XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                          learning_rate=LEARNING_RATE, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    return dv, model

if __name__ == '__main__':
    df = load_data(DATA_PATH)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
    y_full_train = df_full_train['cardio'].values
    y_test = df_test['cardio'].values
    df_full_train = df_full_train.drop(columns='cardio')
    df_test = df_test.drop(columns='cardio')

    dv, model = train(df_full_train, y_full_train)

    X_test = dv.transform(df_test.to_dict(orient='records'))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f'Test AUC: {auc:.4f}')

    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f'Model saved to {OUTPUT_FILE}')