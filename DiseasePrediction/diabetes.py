import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.externals import joblib
import joblib

# DATA FOR PRED
data=pd.read_csv("diabetes.csv")
print(data.head())

logreg=LogisticRegression()
rf=RandomForestClassifier()

X=data.iloc[:,:8]
print(X.shape[1])

y=data[["Outcome"]]

X=np.array(X)
y=np.array(y)

rf.fit(X,y.reshape(-1,))
joblib.dump(rf,"model1")

