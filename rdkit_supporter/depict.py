import matplotlib.pyplot as plt


def feature_importances(model, X, topnum=10):
    topnum = 10
    importances = [
        (fi, name) for fi, name in zip(model.feature_importances_, X.columns)
    ]
    importances = sorted(importances)[-topnum:]
    plt.barh([str(x[1]) for x in importances], [x[0] for x in importances])
    plt.grid()
    plt.show()
