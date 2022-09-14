#Importamos las librerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

#Leyendo datos de prueba y entrenamiento
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

#2.1 Verificar la cantidad de datos que hay en el dataset
#print(df_test.shape)
#print(df_train.shape)

#2.2 Tipos de datos con los que cuenta el dataset
#print(df_train.info())
#print(df_test.info())

#2.3 Datos faltantes
#print(pd.isnull(df_train).sum())
#print("**************************")
#print(pd.isnull(df_test).sum())

#2.4 Estadisticas de cada dataset
#print(df_test.describe())
#print("**************************")
#print(df_train.describe())

#Cambio de los sexos a numero Label encoding
df_train['Sex'].replace(['female','male'],[0,1], inplace = True)
df_test['Sex'].replace(['female','male'],[0,1], inplace = True)  

#Cambio columna de embarque label encoding
df_train['Embarked'].replace(['Q','S','C'],[0,1,2], inplace = True)
df_test['Embarked'].replace(['Q','S','C'],[0,1,2], inplace = True)

#Rellenando datos faltantes en la columna Age
#print(df_train['Age'].mean())
#print(df_test['Age'].mean())

promedio = 30

df_train['Age'] = df_train['Age'].replace(np.nan, promedio)
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)

#Creando bandas de edades
bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)
df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)

#Se elimina cabina porque tiene muchos datos faltantes
df_train.drop(['Cabin'], axis = 1, inplace = True)
df_test.drop(['Cabin'], axis = 1, inplace = True)

#Se eliminan las columnas que no afectan a la predicción final
df_train.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
df_test.drop(['Name', 'Ticket'], axis = 1, inplace = True)

#Se eliminan las filas restantes ya que son muy pocos datos faltantes
df_train.dropna(axis = 0, how = 'any', inplace = True)
df_test.dropna(axis = 0, how = 'any', inplace = True)

#Separamos la columna que será nuestra variable predictiva
X = np.array(df_train.drop(['Survived'],1))
y = np.array(df_train['Survived'])

#Separamos los datos de manera aleatoria
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Regresion logística
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predrg = logreg.predict(X_test)
print("Precision con regresión logística: ")
print(logreg.score(X_train, y_train))

#Maquina de soporte vectorial
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Precision MSV: ")
print(svc.score(X_train, y_train))

#K neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Precision Knn: ")
print(knn.score(X_train, y_train))

total_score_regression = []
total_score_svc = []
total_score_knn = []

for i in range(0,3):
    total_score_regression.append(logreg.score(X_test, y_test))
    total_score_svc.append(svc.score(X_test, y_test))
    total_score_knn.append(knn.score(X_test, y_test))


print("Promedio logistic regression:", np.mean(total_score_regression))
print("Desviación estándar:", np.std(total_score_regression))

print("Promedio Máquina de soporte vectorial:", np.mean(total_score_svc))
print("Desviación estándar:", np.std(total_score_svc))

print("Promedio Knn:", np.mean(total_score_knn))
print("Desviación estándar:", np.std(total_score_knn))

print("Matriz de confusión para la regresión logística")
print(confusion_matrix(y_test, y_predrg))
matrix = confusion_matrix(y_test, y_predrg)
dataframe = pd.DataFrame(matrix)
#sizes, training_scores, testing_scores = learning_curve(LogisticRegression(), X_train, y_train,groups = None,scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 50))

train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(LogisticRegression(max_iter=3000), X, y, cv=30,return_times=True)


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
#plt.show()

sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
#plt.show()

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

logreg2=LogisticRegression(C=1,penalty="l2",)
logreg2.fit(X_train,y_train)
print("score",logreg2.score(X_test,y_test))