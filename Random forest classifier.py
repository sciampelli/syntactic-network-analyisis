#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

RANDOM_SEED=7

print("STARTING")

# ROC
def plot_roc(clf, x_test, y_test):
    print("len(x_test) =", len(x_test))
    print("len(y_test) =", len(y_test))
    # 1 is magic for `pos_label="psychosis"`
    y_proba = clf.predict_proba(x_test)[:,1]
    y_pred = clf.predict(x_test)
    for true,a,b in zip(y_test, y_pred, clf.predict_proba(x_test)):
        print(f'{true}\t{a}\t{b}')
    print("Roc AUC RandomForestClassifier", roc_auc_score(y_test, clf.predict_proba(x_test)[:,1], average='macro'))
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label="psychosis")
    print(f"fpr ({len(tpr)})", fpr)
    print(f"tpr ({len(tpr)})", tpr)
    plt.plot(fpr, tpr, label="rfc")

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

#loading dataset; normalize comes after
dataset = pd.read_csv('output_sentencelength.csv')

#any .csv should work; path can be either absolute or relative to the .py file

#following tutorial; x are features, Y is the "label" or group status. column change depending on csv
y = dataset.iloc[:, 1]
x = dataset.iloc[:, 2:7]
print("how many columns:", x.shape)
N_ESTIMATORS = 100
print("number of estimators:", N_ESTIMATORS)

print(y)
print(x.values)

#divide your test/split set; wih cross-validation (later). test_size gives fraction of data you want to test on.
#recommend to add a random_state number, to aid reproducibility.
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=RANDOM_SEED)

#next step; scaling values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#finally time for the real classifier
classifier = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, criterion='gini')
classifier.fit(x_train, y_train)
print("feature importance")
for feature, importance in zip(dataset.columns[2:7], classifier.feature_importances_):
    print(f'{importance:.3f}\t{feature}')


# Extract single tree
for i, estimator in enumerate(classifier.estimators_):
    pass
    #export_graphviz(estimator, out_file=f'tree_{i}.dot',
                    #feature_names = x.columns,
                    #class_names = np.unique(y),
                    #rounded = True, proportion = False,
                    #precision = 2, filled = True)
    #import os
    #os.system(f"dot -Tpng tree_{i}.dot -o tree_{i}.png ")
# get the ROC plot
#plot_roc(classifier, x_test, y_test)

y_pred = classifier.predict(x_test)
print("confusion list")
print("true\tprediction")
print("----------------")
for true,pred in zip(y_test,y_pred):
    print(f"{true}\t{pred}")
print("----------------")

# initial - non-crossvalidated outcome.
print(confusion_matrix(y_test,y_pred))
print("classification report")
print(classification_report(y_test,y_pred))
print("accuracy score", accuracy_score(y_test, y_pred))


# finally -cross-validated.
forest = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, criterion='gini')

# 10-Fold Cross validation. further values can be gotten from inspecting the outcome of cross_val_score
#validation_score = cross_val_score(forest, x.values, y.values, cv=10)
#print(validation_score)
#outcome = np.mean(validation_score)
#print("outcome", outcome)


###### CROSS_VALIDATED_ROC ########################
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

cv = StratifiedKFold(n_splits = 20)
fig, ax = plt.subplots()

specificity = []
sensitivity = []
accuracy_scores = []

# MANUAL CROSS VALIDATION
for i, (train, test) in enumerate(cv.split(x.values, y.values)):
    forest.fit(x.values[train], y.values[train])
    y_pred = forest.predict(x.values[test])
    print("classification report ", i)
    report = classification_report(y.values[test], y_pred, output_dict=True)
    specificity.append(report['psychosis']['recall'])
    sensitivity.append(report['control']['recall'])
    report = classification_report(y.values[test], y_pred)
    print(report)
    accuracy_scores.append(accuracy_score(y.values[test], y_pred))
    viz = plot_roc_curve(forest, x.values[test], y.values[test],
                         name=f'ROC fold {i}',
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
print("end classification reports")
print(f"accuracy score: {np.mean(accuracy_scores):.2f} ± {np.std(accuracy_scores):.2f}")
print(f"average specificity: {np.mean(specificity):.2f} ± {np.std(specificity):.2f}")
print(f"average sensitivity: {np.mean(sensitivity):.2f} ± {np.std(sensitivity):.2f}")
print("accuracy_scores: ", accuracy_scores)
print("specificity: ", specificity)
print("sensitivity: ", sensitivity)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print("mean AUC", mean_auc)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
ax.get_legend().remove()

# blue_patch = mpatches.Patch(color='blue', label='ROC (AUC) = 0.77')
# red_patch = mpatches.Patch(color='red', label='Chance catzo', linestyle='--')
# plt.legend(handles=[blue_patch, red_patch], loc="lower right")


#Plot red dotted line
lines = [Line2D([0], [0], color='red', linewidth=2, linestyle='--'),
  Line2D([0], [0], color='blue', linewidth=2)]
labels = ['Chance', "ROC (AUC) = 0.69"]
plt.legend(lines, labels, loc = "lower right")
plt.show()


plt.show()
