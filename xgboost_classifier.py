import os
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, precision_score, recall_score
from xgboost import plot_importance

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

# XGBoost Model definition in a pipeline
model = xgb.XGBClassifier(objective="binary:logistic", eval_metric='aucpr', scale_pos_weight=class_ratio, random_state=0)
xgb_pipeline = make_pipeline(StandardScaler(), model)
xgb_pipeline.fit(X_train, y_train)

# Logistic regression model definition in a pipeline
LR_pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', C=10))
LR_pipeline.fit(X_train, y_train)  # apply scaling and training the model

feat_names = X_train.columns
sorted_idx = model.feature_importances_.argsort()

# Plotting feature importance
# plt.barh(feat_names[sorted_idx], model.feature_importances_[sorted_idx])
# plt.savefig('feature_imp_xgboost.png')

y_score_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
y_pred_xgb = xgb_pipeline.predict(X_test)

y_score_LR = LR_pipeline.predict_proba(X_test)[:, 1]
y_pred_LR = LR_pipeline.predict(X_test)

roc_auc_xgb = roc_auc_score(y_test, y_score_xgb)
roc_auc_LR = roc_auc_score(y_test, y_score_LR)
print('ROC AUC xgb: %.5f' % roc_auc_xgb)
print('ROC AUC LR: %.5f' % roc_auc_LR)

# Average precision score
average_precision_xgb = average_precision_score(y_test, y_score_xgb)
average_precision_LR = average_precision_score(y_test, y_score_LR)
print('AP xgb: %.5f' % average_precision_xgb)
print('AP LR: %.5f' % average_precision_LR)

# # === Cross-validation === #
# # evaluate the model
# # It is used RepeatedStratifiedKFold as evaluation method due to the imbalance data
# cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)
# scores = cross_val_score(xgb_pipeline, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# print(scores)
# # summarize performance
# print('Mean ROC AUC: %.5f' % np.mean(scores), np.std(scores))

fig, ax = plt.subplots()
display = PrecisionRecallDisplay.from_estimator(
    xgb_pipeline, X_test, y_test, name="XGBoost classifier", ax=ax
)
PrecisionRecallDisplay.from_estimator(
    LR_pipeline, X_test, y_test, name="LogisticReg", ax=ax
)
_ = display.ax_.set_title("2-class Precision-Recall curve")
plt.show()