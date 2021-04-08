# dataset['hour'] = dataset.hour.str.slice(1, 3).astype(int)
#
# dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
#
#
# plt.suptitle('histgram of numerical columns', fontsize = 20)
#
# for i in range(1, dataset2.shape[1]+1):
#     plt.subplot(3, 4, i)
#
#     f = plt.gca()
#     vals = np.size(dataset.iloc[:, i - 1].unique())
#
#
#     plt.hist(dataset2.iloc[:, i - 1], bins = vals color='red')
#
#
#
#
#
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
#
# classifier = LogisticRegression()
#
#
#
# rfe = RFE(classifier, 20)
# rfe = rfe.fit(X_train, y_train)
#
#
# print(rfe.support_)
#
# pd.concat([pd.DataFrame(X_train.columns[rfe.support_], columns = ["features"]),
#            pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])],
#           axis = 1)
#
#


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



dataset = pd.read_csv('P39-Financial-Data.csv')


dataset.isna().any()


dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedeule', '' ])




dataset2.corrwith(dataset.e_sighed).plot.bar(
    figsize = (20, 29), title = "correlation with E_sighed",
    fontsize=24,
    rot = 34,
    grid=True
)



dataset = dataset.drop(columns=['months_employ'])

dataset['personal_account_monhts'] = (dataset.personal_account_m + (dataset.personal_account_y*12))


dataset[['personal_account_m', 'personal_account_y', 'personal_account_monhts']]


dataset = pd.get_dummies(dataset)

dataset.columns

dataset = dataset.drop(columns=['pay_chedule_semi-mothly'])

response = dataset["e_sighed"]

users = dataset['entry_id']
dataset =


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_train2 = pd.DataFrame(sc_X.transform(X_test))


X_train.index = X_train.index.values
X_test.index = X_test.index.values
X_train.columns = X_train.columns.values
X_train.columns = X_train.columns.values


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state= 0, penalty='l1')

classifier.fit(X_train, y_tranin)


y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score,recall_score
acc = accuracy_score(y_tes, y_pred)
prec = precision_score(y_pred)

result = pd.DataFrame([['Linear Regression']])



from sklearn.svm import SVC

classifier = SVC(random_state= 0, penalty='l1')

classifier.fit(X_train, y_tranin)


y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score,recall_score
acc = accuracy_score(y_tes, y_pred)
prec = precision_score(y_pred)

result = pd.DataFrame([['Linear Regression']])


result = results.append(model_results, ignore_index = True)


#ensemble
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state= 0, n_estimators=100,
                                    criterion = 'entropy',
                                    )
classifier.fit(X_train, y_tranin)


y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score,recall_score
acc = accuracy_score(y_tes, y_pred)
prec = precision_score(y_pred)

result = pd.DataFrame([['Linear Regression']])


result = results.append(model_results, ignore_index = True)

# K-fold Cross Varidation
from sklearn.model_selection import cross_val_score

accuracy_score = cross_val_score(estimator=classifier, X = X_train, y = y_train,
                                 cv = 10)

print("random forest classfier accuracy")


parameters = {"max_depth": [3, None],
              "max_features": [1, 4, 10],
              'min_samples_split': [2, 3, 4],
              'min_samples_leaf': [1, 5, 10],
              'bootstrap': [True, False],
              "criterion": ["entropy"]}
from sklearn.model_selection import GridSearchCV
#gred search
grid_search = GridSearchCV(estimator = classifier,
                           param_grid=parameters,
                           scoding = "accuracy",
                           cv = 10,
                           n_jobs=-1)

import time
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0)





rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_

rf_best_accuracy, rf_best_parameters

y_pred = grid_search.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_result = pd.DataFrame([['Random Forrest (]])

results = results.append(model_result, ignore_index = True)




final_results = pd.concat([y_test, users], axis - 1).dropna()
final_results['predictions'] = y_pred

final_results = final_results[['entry_id', 'e_sighned', 'predictions']]

