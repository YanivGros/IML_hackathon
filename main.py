import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer

if __name__ == '__main__':
    print("start")
    df = pd.read_csv("Mission 2 - Breast Cancer/train.feats.csv")
    # df = df.head(1000)
    df.rename(columns={
        'אבחנה-Age': 'Age',
        'אבחנה-Basic stage': 'Basic stage',
        'אבחנה-Diagnosis date': 'Diagnosis date',
        'אבחנה-Her2': 'Her2',
        'אבחנה-Histological diagnosis': 'Histological diagnosis',
        'אבחנה-Histopatological degree': 'Histopatological diagnosis',
        'אבחנה-Ivi -Lymphovascular invasion': 'Lymphovascular invasion',
        'אבחנה-KI67 protein': 'Lymphovascular invasion',
        'אבחנה-Lymphatic penetration': 'Lymphatic penetration',
        'אבחנה-M -metastases mark (TNM)': 'metastases mark',
        'אבחנה-Margin Type': 'Margin Type',
        'אבחנה-N -lymph nodes mark (TNM)': 'lymph nodes mark',
        'אבחנה-Nodes exam': 'Nodes exam',
        'אבחנה-Positive nodes': 'Positive nodes',
        'אבחנה-Side': 'Side',
        'אבחנה-Stage': 'Stage',
        'אבחנה-Surgery date1': 'Surgery date1',
        'אבחנה-Surgery date2': 'Surgery date2',
        'אבחנה-Surgery date3': 'Surgery date3',
        'אבחנה-Surgery name1': 'Surgery name1',
        'אבחנה-Surgery name2': 'Surgery name2',
        'אבחנה-Surgery name3': 'Surgery name3',
        'אבחנה-Surgery sum': 'Surgery sum',
        'אבחנה-T -Tumor mark (TNM)': 'Tumor mark',
        'אבחנה-Tumor depth': 'Tumor depth',
        'אבחנה-Tumor width': 'Tumor depth',
        'אבחנה-er': 'er',
        'אבחנה-pr': 'pr',
    }, inplace=True)

    label_enc = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    order_enc = preprocessing.LabelEncoder()
    # c = 1, p = 2, r = 3, null = 0
    ordinal_features = ['Basic stage']
    ordinal_orders = [
        # Basic stage
        ['Null', 'c - Clinical', 'p - Pathological', 'r - Reccurent'],
    ]
    transformer = make_column_transformer((OrdinalEncoder(categories=ordinal_orders, handle_unknown="use_encoded_value",
                                                          unknown_value=0),ordinal_features),
                                          remainder="passthrough").fit(df[ordinal_features])

    df[ordinal_features] = transformer.transform(DataFrame(df[ordinal_features]))
    list_to_label = ["Hospital", "Form Name"]
    label_enc.fit(df[list_to_label])
    df[list_to_label] = label_enc.transform(df[list_to_label])

    hot_enc = preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")
    list_to_hot_encode = ["Form Name", "Hospital"]
    hot_enc.fit(df[list_to_hot_encode])
    dfOneHot = DataFrame(hot_enc.transform(df[hot_enc.feature_names_in_]))
    dfOneHot.columns = hot_enc.get_feature_names_out()
    df = df.join(dfOneHot)
    df.info()
    print("end")

    # Order_of_basic_stage = [
    #     ['Null'],
    #     ['c - Clinical'],
    #     ['p - Pathological'],
    #     ['r - Reccurent'],
    # ]
    # X = [
    #     ['c - Clinical', 1],
    #     ['p - Pathological', 2],
    #     ['r - Reccurent', 3],
    #     ['Null', 0],
    # ]
