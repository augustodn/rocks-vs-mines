import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import layers
from keras import Sequential
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from xgboost import XGBClassifier

sonar = pd.read_csv('sonar.all-data.csv', header=None)
sonar.head()

corr = sonar.corr()
sns.heatmap(corr)

# There is some correlation between some variables, for so the dataset is
# candidate for dimensionality reduction

X = sonar.loc[:, :59]
y = sonar.loc[:, 60]
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, shuffle=True)

encoder = load_model('encoder.h5')

X_encode = encoder.predict(X)
X_train_encode = encoder.predict(X_train)
X_test_encode = encoder.predict(X_test)
n_obs, n_inputs = X_train_encode.shape

models = [
    ('LogReg', LogisticRegression()),
    ('DT', tree.DecisionTreeClassifier()),
    ('RF', RandomForestClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(probability=True)),
    ('GNB', GaussianNB()),
    ('XGB', XGBClassifier(verbosity=0, use_label_encoder=False))
]

target_names = le.classes_

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
           'roc_auc']
results = []
names = []
model_cv_results = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=1986)
    cv_result = model_selection.cross_validate(model, X_train_encode, y_train,
                                               cv=kfold, scoring=scoring)

    results.append(cv_result)
    names.append(name)

    df = pd.DataFrame(cv_result)
    df['model'] = name
    model_cv_results.append(df)

model_cv_results = pd.concat(model_cv_results, ignore_index=True)
print(model_cv_results.groupby(['model']).mean())


acc = list()
training = [(X_train, X_test),
            (X_train_encode, X_test_encode)]

for x_train, x_test in training:

    model = LogisticRegression()
    model.fit(x_train, y_train)
    yhat = model.predict(x_test)

    acc.append(accuracy_score(y_test, yhat))

print(acc)

# Model a Binary classifier
model = Sequential()
model.add(layers.Dense(12, input_dim=n_inputs, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# Define optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# Train the model
history = model.fit(X_train_encode, y_train, epochs=500, batch_size=n_obs,
                    validation_data=(X_test_encode, y_test), verbose=0)
print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='test')
axes[0].legend()
axes[0].title.set_text('Loss')
axes[1].plot(history.history['accuracy'], label='train')
axes[1].plot(history.history['val_accuracy'], label='test')
axes[1].title.set_text('Accuracy')
axes[1].legend()

plt.show()

# Model a Binary classifier
model = Sequential()
model.add(layers.Dense(60, input_dim=60, activation='relu'))
model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# Define optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# Train the model


history = model.fit(X_train, y_train, epochs=500, batch_size=n_obs,
                    validation_data=(X_test, y_test), verbose=0)
print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")
