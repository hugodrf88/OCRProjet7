import joblib

# création d'une pipeline qui reprend le processeur précédent
# avec la régression logistique
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from category_encoders import TargetEncoder
from functions.functions import fill_cat,fill_num
from sklearn.preprocessing import OneHotEncoder

data=pd.read_csv("../application_train.csv")
# gestion du déséquilibre de classe
X_1=data[data["TARGET"]==1]
X_0=data[data["TARGET"]==0]
X_0=X_0.sample(n=X_1.shape[0])
X= pd.concat([X_0, X_1]).sample(frac=1).reset_index(drop=True)

from sklearn.model_selection import train_test_split

train,validation=train_test_split(X,test_size=0.25,random_state=64,stratify=X.TARGET
 )
#test,validation=train_test_split(test,test_size=0.5,random_state=123,stratify=test.TARGET)
from sklearn.model_selection import train_test_split

X_train,y_train=train.iloc[:,1:],train.iloc[:,0]
X_val,y_val=validation.iloc[:,1:],validation.iloc[:,0]


num_cols=data.select_dtypes(exclude=["object"]).columns.tolist()
num_cols=[col for col in num_cols if col!="TARGET"]
cat_cols=data.select_dtypes(include=["object"]).columns.tolist()


num_transformer=Pipeline(steps=[("num_imputer",FunctionTransformer(func=fill_num,kw_args={"strategy":"mean"})),
                                ("scaler",StandardScaler())])
cat_transformer=Pipeline(steps=[("cat_imputer",FunctionTransformer(func=fill_cat,kw_args={"most_frequent":False})),
                                 ("encoder",TargetEncoder())])

preprocessor=ColumnTransformer(
transformers=[
    ("num",num_transformer,num_cols),
    ("cat",cat_transformer,cat_cols)
])



seed = 123
pipeline_model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('logreg', LogisticRegression(random_state=seed))])

# Définir les hyperparamètres à explorer
param_distributions = {'logreg__C': loguniform(1e-4, 1e4),
                       'logreg__penalty': [None, 'l2'],
                       'logreg__solver': ['liblinear', 'lbfgs', 'saga', 'sag', 'newton-cg'],
                       'logreg__max_iter': [100, 500, 1000, 2000],
                       'preprocessor__num__num_imputer__kw_args': [{"strategy": "mean"}, {"strategy": "median"}],
                       'preprocessor__cat__cat_imputer__kw_args': [{"most_frequent": True}, {"most_frequent": False}]

                       }

# Créer un objet LogisticRegression() et un objet RandomizedSearchCV()

search_model = RandomizedSearchCV(estimator=pipeline_model, param_distributions=param_distributions, n_iter=50, cv=5,
                                  scoring='roc_auc', verbose=2, n_jobs=-1)
print(y_train.value_counts())
search_model.fit(X_train, y_train)
print(search_model.best_params_)
print(search_model.best_score_)
# enregistrement du meilleur modè.le
joblib.dump(search_model.best_estimator_,'./models/best_model.pkl')