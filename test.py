import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC

DGA = 1
NOTDGA = 0

def entropy(str):
    result = {}
    length = len(str)
    for c in np.unique(list(str)):
        result[c] = str.count(c)/length
    ety = 0
    for key in result:
        ety -= result[key] * np.log2(result[key])
    return ety

def numofdigit(str):
    num_digit = 0
    for c in str:
        if c.isdigit():
            num_digit += 1
    return num_digit

def domainFeature(domain):
    return len(domain), numofdigit(domain), entropy(domain)


class Domain:
    def __init__(self, _domain, _label):
        self.feature = list(domainFeature(_domain))
        self.label = _label

    def getFeature(self):
        return self.feature

    def getLabel(self):
        return self.label

# Process the training data into feature matrix and
# corresponding label
train_data = pd.read_csv('data/train.txt', sep=',', names=['domain', 'label'])
test_data = pd.read_csv('data/test.txt', sep=',', names=['domain'])
featureMatrix = []
label = []
for index, row in train_data.iterrows():
    domain = Domain(row['domain'], row['label'])
    featureMatrix.append(domain.getFeature())
    label.append(DGA if domain.getLabel() == 'dga' else NOTDGA)

clf = RFC()
clf.fit(featureMatrix, label)

testFeatureMatrix = []


for index, row in test_data.iterrows():
    testFeatureMatrix.append(list(domainFeature(row['domain'])))

tmpLabel = clf.predict(testFeatureMatrix)

testLabel = ['notdga' if l == NOTDGA else 'dga' for l in tmpLabel]
test_data['label'] = testLabel

test_data.to_csv('result.txt', sep=',')