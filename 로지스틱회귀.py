import pandas as pd

passengers = pd.read_csv("train.csv")

print(passengers.shape)
print(passengers.head())

df = passengers
corr = df.corr(method='pearson')
print(corr)

# fare pclass age 3개의 특성을 선택
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x : 1 if x == 1 else 0)
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x : 2 if x == 2 else 0)

passengers['Age'].fillna(value = passengers['Age'].mean(), inplace = True)

features = passengers[['FirstClass', 'SecondClass', 'Fare', 'Age']]
survival = passengers['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, survival)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))

print(model.score(X_test, y_test))

print(model.coef_)
