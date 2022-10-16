import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestRegressor

def feature_select(
    df,
    X_cols,
    y_col,
    n_estimators=1000,
    max_features="sqrt", 
):
    estimator = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_features="sqrt", 
    n_jobs=-1)
    X = df[X_cols]
    y = df[y_col]

    selector = RFECV(estimator=estimator, n_features_to_select=30, verbose=2)
    selector.fit(X, y)

    print("N_features %s" % selector.n_features_)
    print("Ranking %s" % selector.ranking_)
    return selector.support_
