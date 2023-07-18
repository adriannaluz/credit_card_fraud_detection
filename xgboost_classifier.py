import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.pipeline import make_pipeline
from sklearn.metrics import auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, precision_score, recall_score, RocCurveDisplay
# from xgboost import plot_importance

warnings.filterwarnings('ignore')


# loading .env file
load_dotenv()

data_path = os.getenv("data_path")
file = "creditcard.csv"
data = pd.read_csv(data_path + file)

# Counting the data in each class
class_count = data.Class.value_counts() / len(data.Class.index)
# print(f"Percentage of data in each class: \n{class_count}")

# class ratio
class_ratio = data.Class[data.Class == 0].size / data.Class[data.Class == 1].size
# print(f"Percentage of data in each class: \n{class_ratio}")

y = data["Class"]
train = data.copy().drop("Class", axis=1)

# Training, Validation and testing data splitting
X_train, X_test, y_train, y_test = train_test_split(
    train, y, train_size=0.8, test_size=0.2, random_state=0
)

# XGBoost Model definition
# max_delta_step for imbalance data
model = xgb.XGBClassifier(
    scale_pos_weight=class_ratio,
    eval_metric="aucpr",
    max_delta_step=1,
    tree_method="hist",
    random_state=0,
    gamma=2,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1
)

# Definition of the pipeline
pipeline = make_pipeline(StandardScaler(), xgb.XGBClassifier(objective='binary:hinge'))

# Print parameters to fine tune
# print(pipeline.get_params().keys())

# # Parameter for gridsearch
# params = {}
# params["xgbclassifier__learning_rate"] = [0.3, 0.1, 0.03, 0.01, 0.003]
# params["xgbclassifier__max_depth"] = [3, 6]
# params["xgbclassifier__min_child_weight"] = [1, 5, 10]
# params["xgbclassifier__gamma"] = [1, 2, 4]
#
# # # Train the grid search model
# gs = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
# print(f"Best input parameters: {gs.best_params_}")
# # Best input parameters: {'xgbclassifier__gamma': 2, 'xgbclassifier__learning_rate': 0.1, 'xgbclassifier__max_depth': 6, 'xgbclassifier__min_child_weight': 1}
# print(f"Best score: {gs.best_score_}")
# # Best score: 0.9801892785651803

# === Cross-validation === #
# evaluate the model
# It is used RepeatedStratifiedKFold as evaluation method due to the imbalance data
n_splits = 5
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=0)

# Train the classifier and get the training history
history = []
X = X_train.copy()
y = y_train.copy()
for train_index, val_index in cv.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Fit the classifier with eval_set
    eval_set = [(X_val, y_val)]
    pipeline[1].fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Get the training history for this fold
    fold_history = pipeline[1].evals_result()
    history.append(fold_history)

# Extract the loss values from the training history
train_loss = history
print(train_loss)
# Plot the loss function
fig, ax = plt.subplots()
for fold, loss in enumerate(train_loss):
    ax.plot(loss['validation_0']['error'], label=f'Fold {fold+1}')

ax.set_title('XGBoost Classifier Loss')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.legend()
plt.show()


# scores = cross_val_score(xgb_pipeline, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# print(scores)
# # summarize performance
# print('Mean ROC AUC: %.5f' % np.mean(scores), np.std(scores))
