import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def normalize(data_frame):
    values = data_frame.drop(['class'], axis=1)
    classes = data_frame[['class']]
    values = (values - values.mean())/values.std()
    values['class'] = classes['class']
    return values


data = pd.read_excel('data.xlsx')
data_name = data[['name']]
data = data.drop(['name'], axis=1)
data.fillna(0, inplace=True)
data['class'] = data['class'].astype('int32')
data = normalize(data)

learning_sample = data.loc[data['class'] != 0]
prediction_sample = data.loc[data['class'] == 0].drop(['class'], axis=1)

clf = LinearDiscriminantAnalysis()
x = np.array(learning_sample.drop(['class'], axis=1))
y = np.array(learning_sample['class'])
clf.fit(x, y)

predicted_classes = clf.predict(np.array(prediction_sample))
prediction_sample['class'] = predicted_classes

result = prediction_sample.append(learning_sample)
result['name'] = data_name['name']
result = result.sort_index()

result.to_excel('result.xlsx', sheet_name='result')
