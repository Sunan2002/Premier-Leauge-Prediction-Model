from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

df_last = pd.read_csv('Prem_21_22.csv')
df_next = pd.read_csv('Prem_22_23.csv')

test_size = 200

'''
TRAINING SET TARGET 
'''
# setup target -> home team wins vs ties/loses
y_train = np.array([])

# for i in range(len(df_last)):
for i in range(test_size):
  if df_last['FTHG'][i] > df_last['FTAG'][i]: y_train = np.append(y_train, 1)
  else: y_train = np.append(y_train, 0)

'''
TEST SET TARGET
'''
y_test = np.array([])

for i in range(test_size):
  if df_next['FTHG'][i] > df_next['FTAG'][i]: y_test = np.append(y_test, int(1))
  else: y_test = np.append(y_test, int(0))


'''
TRAINING SET FEATURE(S)
'''
# setup feature(s) -> Goal difference at halftime (home - away)
X_train = np.array([])

# for i in range(len(df_last)):
for i in range(test_size):
  X_train = np.append(X_train, df_last['HTHG'][i] - df_last['HTAG'][i])

'''
TEST SET FEATURES(S)
'''
X_test = np.array([])

for i in range(test_size):
  X_test = np.append(X_test, df_next['HTHG'][i] - df_next['HTAG'][i])


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train.reshape(1, -1), y_train.reshape(1, -1))

pred = np.array(clf.predict(X_test.reshape(1,-1))[0])

a = 0
for i in range(test_size):
  if pred[i] == y_test[i]: a += 1

print('Accuracy: {}%'.format((a/test_size)*100))