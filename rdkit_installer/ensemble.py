import glob
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit_installer import preprocess
from rdkit_installer.descriptors import calc_descriptors
from rdkit_installer.fingerprints import Fingerprinter
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor, MLPClassifier


class SmilesBaggingMLP:
    def __init__(self, smiles_col, target_col, n_samples=1000, estimator=MLPRegressor):
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.n_samples = n_samples
        self.estimator = estimator

    def train(self, data_df, trained_model_path, max_trial=10, recording_threshold=0.9):
        for trial in range(max_trial):
            indexs = random.choices(range(data_df.shape[0]), k=self.n_samples)
            bagged_data = data_df.iloc[indexs, :]

            fingerprinter = Fingerprinter()
            x_choice = random.choice([x for x in range(len(fingerprinter.names) + 5)])
            name_x = "rdkit"
            if x_choice < len(fingerprinter.names):
                name_x = fingerprinter.names[x_choice]
                fp_type = name_x
                X_df = pd.DataFrame(
                    [
                        v
                        for v in fingerprinter.transform(
                            bagged_data[self.smiles_col], fp_type=fp_type
                        )
                    ]
                )
                self.n_columns = 1000
            else:
                X_df = calc_descriptors(bagged_data[self.smiles_col])
                self.n_columns = 200

            cleaner = preprocess.TableCleaner()
            success_col = cleaner.clean_columns(X_df, bagged_data[self.target_col])
            X_df = X_df.iloc[:, success_col]
            X_df = preprocess.remove_low_variance_features(X_df)
            colids = random.choices(range(X_df.shape[1]), k=self.n_columns)
            X_df = X_df.iloc[:, colids]
            X_df = preprocess.remove_high_correlation_features(X_df, threshold=0.95)

            print(name_x)
            model = self.estimator(
                hidden_layer_sizes=random.choice(
                    [
                        (100),
                        (100, 100),
                        (100, 100, 100),
                    ]
                ),
                max_iter=530000,
                learning_rate=random.choice(["constant", "invscaling", "adaptive"]),
                activation=random.choice(["logistic", "tanh", "relu"]),
                early_stopping=random.choice([True, False]),
            )
            model.fit(X_df, bagged_data[self.target_col])
            score = model.score(X_df, bagged_data[self.target_col])
            if score == 1.0:
                score += random.random() * 1e-8
            print(score, model)
            if score >= recording_threshold:
                with open(
                    "{}model_{}.pkl".format(trained_model_path, score), "wb"
                ) as f:
                    pickle.dump(model, f)

                with open(
                    "{}column_{}.pkl".format(trained_model_path, score), "wb"
                ) as f:
                    pickle.dump({"columns": list(X_df.columns), "type": name_x}, f)

    def load(self, trained_model_path):
        self.models = []
        models_paths = glob.glob(trained_model_path + "model*")
        for model_path in models_paths:
            id = model_path.split("_")[1]
            columns_path = trained_model_path + "column_" + id
            if os.path.isfile(columns_path):
                score = float(".".join(id.split(".")[:-1]))
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                with open(columns_path, "rb") as f:
                    columns = pickle.load(f)
                self.models.append([score, model, columns["columns"], columns["type"]])

    def predict(self, data_df):
        return self.tune_predict(data_df, tuning=False)

    def tune_predict(self, data_df, tuning=True):
        Y_df = pd.DataFrame([])
        iter = 0
        while True:
            start_sample = iter * self.n_samples
            finish_sample = (iter + 1) * self.n_samples
            if start_sample > data_df.shape[0]:
                break
            if finish_sample > data_df.shape[0]:
                finish_sample = data_df.shape[0]
            Ys = []
            scores = []
            X_df = {}
            X_df["rdkit"] = calc_descriptors(
                data_df.iloc[start_sample:finish_sample, :][self.smiles_col]
            )
            fingerprinter = Fingerprinter()
            for fp_type in fingerprinter.names:
                X_df[fp_type] = pd.DataFrame(
                    [
                        v
                        for v in fingerprinter.transform(
                            data_df.iloc[start_sample:finish_sample, :][
                                self.smiles_col
                            ],
                            fp_type=fp_type,
                        )
                    ]
                )

            for score, model, columns, fp_type in sorted(self.models, reverse=True):
                print(score, model, columns)

                try:
                    Y = model.predict(X_df[fp_type][columns])
                except KeyError:
                    Y = np.array([np.nan for x in range(X_df[fp_type].shape[0])])
                except ValueError:
                    Y = np.array([np.nan for x in range(X_df[fp_type].shape[0])])

                Ys.append(Y)
                scores.append(score)

            y_df = pd.DataFrame(np.matrix(Ys).T)
            y_df.columns = scores
            Y_df = pd.concat([Y_df, y_df], ignore_index=True)
            iter += 1

        self.Y_df = Y_df.dropna(axis=1)

        if tuning:
            selected_col = []
            best_score = 0
            for i in range(self.Y_df.shape[1]):
                selected_col_copy = [x for x in selected_col]
                selected_col_copy.append(i)
                score = r2_score(
                                    data_df[self.target_col],
                                    self.Y_df.iloc[:, selected_col_copy].mean(axis=1))
                if best_score < score:
                    best_score = score
                    selected_col = selected_col_copy

            self.selected_col = selected_col
            print("ensemble ", len(selected_col), "models,")
            print(list(self.Y_df.iloc[:, selected_col].columns))

        return (
            self.Y_df.iloc[:, self.selected_col].mean(axis=1).values,
            self.Y_df.iloc[:, self.selected_col].std(axis=1).values,
        )