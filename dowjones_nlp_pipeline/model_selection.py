from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import graphviz
import os

class ModelSelector:
    def __init__(self, X, y, base_ticker, random_iter=20):
        """
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series or np.ndarray
            Target labels.
        base_ticker : str
            The ticker symbol from config.BASE_TICKER.
        random_iter : int
            Number of iterations for RandomizedSearchCV.
        """
        self.X = X
        self.y = y
        self.base_ticker = base_ticker
        self.random_iter = random_iter
        self.tuned_models = {}
        self.results = {}
        self.sorted_models = []
        self.top_two = []

    def tune_models(self):
        # Split for tuning/validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y,
            test_size=0.10,
            random_state=123,
            stratify=self.y
        )
        self.X_val, self.y_val = X_val, y_val

        # -------------------------
        # Decision Tree
        # -------------------------
        dt_random_params = {
            'max_depth': [None, 2, 3, 5, 7, 10],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 6]
        }
        dt_random = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=123),
            dt_random_params,
            n_iter=self.random_iter,
            cv=5,
            scoring='accuracy',
            random_state=123,
            n_jobs=-1
        )
        dt_random.fit(X_train, y_train)

        best_dt = dt_random.best_params_
        dt_grid_params = {
            'max_depth': [best_dt['max_depth'] - 1 if best_dt['max_depth'] else None,
                          best_dt['max_depth'],
                          best_dt['max_depth'] + 1 if best_dt['max_depth'] else None],
            'min_samples_split': [max(2, best_dt['min_samples_split'] - 1),
                                  best_dt['min_samples_split'],
                                  best_dt['min_samples_split'] + 1],
            'min_samples_leaf': [max(1, best_dt['min_samples_leaf'] - 1),
                                 best_dt['min_samples_leaf'],
                                 best_dt['min_samples_leaf'] + 1]
        }
        dt_grid = GridSearchCV(
            DecisionTreeClassifier(random_state=123),
            dt_grid_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        dt_grid.fit(X_train, y_train)
        self.tuned_models['DecisionTree'] = dt_grid.best_estimator_

        # -------------------------
        # Random Forest
        # -------------------------
        rf_random_params = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_random = RandomizedSearchCV(
            RandomForestClassifier(random_state=123),
            rf_random_params,
            n_iter=self.random_iter,
            cv=5,
            scoring='accuracy',
            random_state=123,
            n_jobs=-1
        )
        rf_random.fit(X_train, y_train)

        best_rf = rf_random.best_params_
        rf_grid_params = {
            'n_estimators': [max(50, best_rf['n_estimators'] - 50),
                             best_rf['n_estimators'],
                             best_rf['n_estimators'] + 50],
            'max_depth': [best_rf['max_depth'] - 1 if best_rf['max_depth'] else None,
                          best_rf['max_depth'],
                          best_rf['max_depth'] + 1 if best_rf['max_depth'] else None],
            'min_samples_split': [max(2, best_rf['min_samples_split'] - 1),
                                  best_rf['min_samples_split'],
                                  best_rf['min_samples_split'] + 1],
            'min_samples_leaf': [max(1, best_rf['min_samples_leaf'] - 1),
                                 best_rf['min_samples_leaf'],
                                 best_rf['min_samples_leaf'] + 1]
        }
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=123),
            rf_grid_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        self.tuned_models['RandomForest'] = rf_grid.best_estimator_

        # -------------------------
        # Gradient Boosting
        # -------------------------
        gb_random_params = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [2, 3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2]
        }
        gb_random = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=123),
            gb_random_params,
            n_iter=self.random_iter,
            cv=5,
            scoring='accuracy',
            random_state=123,
            n_jobs=-1
        )
        gb_random.fit(X_train, y_train)

        best_gb = gb_random.best_params_
        gb_grid_params = {
            'n_estimators': [max(50, best_gb['n_estimators'] - 50),
                             best_gb['n_estimators'],
                             best_gb['n_estimators'] + 50],
            'max_depth': [max(2, best_gb['max_depth'] - 1),
                          best_gb['max_depth'],
                          best_gb['max_depth'] + 1],
            'learning_rate': [max(0.01, best_gb['learning_rate'] / 2),
                              best_gb['learning_rate'],
                              best_gb['learning_rate'] * 2]
        }
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=123),
            gb_grid_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        self.tuned_models['GradientBoosting'] = gb_grid.best_estimator_

    def compare_models(self):
        # Evaluate tuned models
        for name, model in self.tuned_models.items():
            preds = model.predict(self.X_val)
            acc = accuracy_score(self.y_val, preds)
            prec = precision_score(self.y_val, preds, zero_division=0)
            mse = mean_squared_error(self.y_val, preds)
            self.results[name] = {"Accuracy": acc, "Precision": prec, "MSE": mse}

            if name == "DecisionTree":
                # Save tree with ticker in filename
                dot_filename = f"tree_{self.base_ticker}.dot"
                export_graphviz(
                    model,
                    out_file=dot_filename,
                    class_names=["Down", "Up"],
                    feature_names=self.X.columns,
                    impurity=False,
                    filled=True
                )
                with open(dot_filename) as file:
                    decision_tree = file.read()
                graphviz.Source(decision_tree)

        # Rank models by combined score
        mse_values = np.array([m["MSE"] for m in self.results.values()]).reshape(-1, 1)
        mse_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(mse_values)
        mse_scaled = 1 - mse_scaled

        combined_scores = {}
        for (name, metrics), mse_score in zip(self.results.items(), mse_scaled):
            combined_scores[name] = (metrics["Accuracy"] + metrics["Precision"] + mse_score[0]) / 3

        self.sorted_models = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        self.top_two = self.sorted_models[:2]

        return self.results, self.sorted_models, self.top_two
