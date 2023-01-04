import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.pyplot as plt

def regression_metrics(model, X, Y):
    Y_pred = model.predict(X)
    scores = []
    scores.append(["MedAE", metrics.median_absolute_error(Y_pred, Y)])
    scores.append(["MAE", metrics.mean_absolute_error(Y_pred, Y)])
    scores.append(["RMSE", metrics.mean_squared_error(Y_pred, Y, squared=False)])
    r2 = metrics.r2_score(Y, Y_pred)
    y_max = max(Y.max(), Y_pred.max())
    y_min = min(Y.min(), Y_pred.min())
    y_height = abs(y_max - y_min) / 2
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    axes[0].barh([x[0] for x in scores], [x[1] for x in scores])
    axes[0].set_xlabel("Score")
    axes[0].grid()
    axes[1].set_title("R2={}".format(r2))
    axes[1].scatter(Y, Y_pred, alpha=0.5, c=Y)
    axes[1].plot([y_min, y_max], [y_min, y_max], c="r")
    axes[1].set_xlabel("Y_true")
    axes[1].set_ylabel("Y_pred")
    axes[1].grid()
    axes[1].set_aspect ('equal')
    axes[2].set_title("R2={}".format(r2))
    axes[2].scatter(Y, Y_pred - Y, alpha=0.5, c=Y)
    axes[2].plot([y_min, y_max], [0, 0], c="r")
    axes[2].set_xlabel("Y_true")
    axes[2].set_ylabel("Y_err")
    axes[2].set_ylim([-y_height, y_height])
    axes[2].grid()
    plt.show()
    
    
def classification_metrics(model, X, Y):
    Y_pred = model.predict(X)
    scores = []
    scores.append(["MCC", metrics.matthews_corrcoef(Y_pred, Y)])
    scores.append(["Cohen's Kappa", metrics.cohen_kappa_score(Y_pred, Y)])
    scores.append(["F1", metrics.f1_score(Y_pred, Y)])
    scores.append(["Average Precision", metrics.average_precision_score(Y_pred, Y)])
    scores.append(["Precision", metrics.precision_score(Y_pred, Y)])
    scores.append(["Recall", metrics.recall_score(Y_pred, Y)])
    scores.append(["AUC", metrics.balanced_accuracy_score(Y_pred, Y)])
    #scores.append(["TopK ACC", metrics.top_k_accuracy_score(Y_pred, Y)])
    scores.append(["Balanced ACC", metrics.balanced_accuracy_score(Y_pred, Y)])
    scores.append(["ACC", metrics.accuracy_score(Y_pred, Y)])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].barh([x[0] for x in scores], [x[1] for x in scores])
    axes[0].set_xlim([0, 1])
    axes[0].set_xlabel("Score")
    axes[0].grid()

    tn, fp, fn, tp = metrics.confusion_matrix(Y, Y_pred).ravel()
    axes[1].set_title("tn, fp, fn, tp = {} {} {} {}".format(tn, fp, fn, tp))
    axes[1].bar(["Positive", "Negative"], [tp, tn])
    axes[1].bar(["Positive", "Negative"], [-fn, -fp])
    axes[1].grid()
    plt.show()
    
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn import metrics
import sklearn.svm._classes

import copy
def kfold_cv(model, X, Y, measure=r2_score):
    n_splits=4
    kf = KFold(n_splits=n_splits, random_state=53, shuffle=True)
    fig, axes = plt.subplots(nrows=1, ncols=n_splits, figsize=(4*n_splits, 4))
    models = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index, :]
        X_test = X.iloc[test_index, :]
        Y_train = Y.iloc[train_index]
        Y_test = Y.iloc[test_index]
        model.fit(X_train, Y_train)
        models.append(copy.deepcopy(model))
            
        Y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba") or type(model) is sklearn.svm._classes.SVC:
            tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
            axes[i].set_title("tn, fp, fn, tp = {} {} {} {}".format(tn, fp, fn, tp))
            axes[i].bar(["Positive", "Negative"], [tp, tn])
            axes[i].bar(["Positive", "Negative"], [-fn, -fp])
            axes[i].grid()
        else:
            mae = measure(Y_test, Y_pred)
            y_max = max(Y.max(), Y_pred.max())
            y_min = min(Y.min(), Y_pred.min())
            axes[i].set_title("R2={}".format(mae))
            axes[i].scatter(Y_test, Y_pred, alpha=0.5, c=Y_test)
            axes[i].plot([y_min, y_max], [y_min, y_max], c="r")
            axes[i].set_xlabel("Y_true")
            if i == 0:
                axes[i].set_ylabel("Y_pred")
            axes[i].grid()
    plt.show()
    return models

def feature_importances(model, X, topnum=10):
    topnum = 10
    importances = [
        (fi, name) for fi, name in zip(model.feature_importances_, X.columns)
    ]
    importances = sorted(importances)[-topnum:]
    plt.barh([str(x[1]) for x in importances], [x[0] for x in importances])
    plt.grid()
    plt.show()
