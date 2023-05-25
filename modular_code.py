
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import logging
import joblib


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay,classification_report
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split




logging.basicConfig(
    filename='./logs/modular_code.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
    )

def import_data(pth):
    df=pd.read_csv(pth)
    df=df.drop("SK_ID_CURR",axis=1)

    return df

def data_splitting(df):
    """
    Fonction pour séparer le jeu de données en train et test sets avec équilibrage des classes.
    
    Paramètres:
    -----------
    df : pandas DataFrame
        Le jeu de données à diviser en train et test sets.
        
    Retours:
    --------
    train_df : pandas DataFrame
        Le train set équilibré.
    test_df : pandas DataFrame
        Le test set équilibré.
    """
    # gestion du déséquilibre de classe
    X_1 = df[df["TARGET"]==1]
    X_0 = df[df["TARGET"]==0]
    X_0 = X_0.sample(n=X_1.shape[0])
    X = pd.concat([X_0, X_1]).sample(frac=1).reset_index(drop=True)
    
    # division du jeu de données en train et test sets avec équilibrage des classes
    train_df, test_df = train_test_split(X, test_size=0.2, random_state=64, stratify=X.TARGET)
    
    # sauvegarde des différents ensembles de données
    train_df.to_csv("./data/train.csv",index=False)
    test_df.to_csv("./data/test.csv",index=False)
    
    X_train,y_train=train_df.iloc[:,1:],train_df.iloc[:,0]
    X_test,y_test=test_df.iloc[:,1:],test_df.iloc[:,0]
    
    return train_df,X_train,X_test,y_train,y_test


def perform_eda(df):
    """
    perform eda on df and save figures to images
    folder
    input : 
            df_path : a path to csv data
    output : 
            None        

    """
    df_copy=df.copy()
    list_columns=df_copy.columns.tolist()
    list_columns.append("Heatmap")
    list_columns.append("Numerical")
    num_cols=df_copy.select_dtypes(exclude=["object"]).columns.tolist()

    df_corr=df_copy.corr(numeric_only=True)
    
    for column_name in list_columns:
        plt.figure(figsize=(10,6))
        if column_name=="Heatmap":
            plt.figure(figsize=(120, 100))
            sn.heatmap(df_corr, annot=True, cmap="coolwarm")
            plt.savefig("./images/eda/"+column_name+".jpg")
            plt.close()
            
        elif column_name=="Numerical":
                   # Créer une figure plus grande
            fig, axes = plt.subplots(figsize=(40,32))
            
            # Tracer l'histogramme des variables numériques sur la figure
            df_copy[num_cols].hist(ax=axes,bins=20)
            
            # Ajouter un titre à la figure
            fig.suptitle('Histogrammes des variables numériques', fontsize=22)
            
            # Afficher la figure
            plt.savefig("./images/eda/"+column_name+".jpg")
            plt.close()
            
        elif df_copy[column_name].dtype=='O':
            df_copy[column_name].hist()
            plt.savefig("./images/eda/"+column_name+".jpg")
            plt.close()
    
def classification_report_image(y_tr,y_tr_pred,y_t,y_t_pred):
    """
    produces classification report for training and testing results and stores report as
    image in images folder
    input:
            y_tr : training response values
            y_tr_pred : training predictions from logistic regression
            y_t : test response values
            y_t_pred : test predictions from logistic regression
            
    output : None
    """
    
    class_reports_dico={
        "Logistic Regression train results":classification_report(
            y_tr,y_tr_pred),
        "Logistic Regression test results": classification_report(
            y_t,y_t_pred)}
    
    for title, report in class_reports_dico.items():
        plt.rc("figure",figsize=(7,3))
        plt.text(
            0.2,0.3,str(report),{
                "fontsize":10},fontproperties="monospace")
        plt.axis("off")
        plt.title(title,fontweight="bold")
        plt.savefig("./images/results"+title+".jpg")
        plt.close()
        
        


def fill_cat(df, most_frequent=False):
    # Copie du DataFrame pour éviter toute modification de l'original
    df_tmp = df.copy()
    
    # Sélection des colonnes catégorielles dans le DataFrame
    # et stockage dans une liste
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']

    # Remplissage des valeurs manquantes dans chaque colonne catégorielle
    # en utilisant la stratégie "most_frequent" de SimpleImputer
    if most_frequent: 
        imputer = SimpleImputer(strategy='most_frequent')
        # Application de la méthode fit_transform de SimpleImputer 
        # pour remplacer les valeurs manquantes par la valeur la plus fréquente
        df_tmp[cat_cols] = imputer.fit_transform(df_tmp[cat_cols])
    # Sinon, remplir les valeurs manquantes dans chaque colonne catégorielle 
    # avec la chaîne "Other"
    else:
        df_tmp[cat_cols] = df_tmp[cat_cols].fillna("Other")

    # Retourne les valeurs du DataFrame rempli
    return df_tmp.values


def fill_num(df, strategy="median"):

    """
    Remplit les valeurs manquantes dans les colonnes numériques d'un DataFrame avec une stratégie donnée.

    Args:
        df (pandas.DataFrame): DataFrame contenant les colonnes numériques à remplir.
        strategy (str): Stratégie à utiliser pour remplir les valeurs manquantes. Les options sont "mean",
            "median", "most_frequent" et "constant". La valeur par défaut est "median".

    Returns:
        numpy.ndarray: Un tableau NumPy contenant les colonnes numériques avec les valeurs manquantes remplies.

    """
    # Copier le DataFrame d'entrée pour éviter de modifier l'original
    df_tmp = df.copy()

    # Créer un objet SimpleImputer avec la stratégie spécifiée
    imputer = SimpleImputer(strategy=strategy)

    # Sélectionner les colonnes numériques dans le DataFrame
    cols = [col for col in df.columns if df[col].dtype == "float64"]

    # Remplacer les valeurs manquantes dans chaque colonne numérique
    df_tmp[cols] = imputer.fit_transform(df_tmp[cols])

    # Retourner les colonnes numériques avec les valeurs manquantes remplies sous forme de tableau NumPy
    return df_tmp.values


def build_pipeline(df,X_tr,y_tr):
    y_tr=y_tr.reset_index(drop=True)

    num_cols=df.select_dtypes(exclude=["object"]).columns.tolist()
    num_cols=[col for col in num_cols if col!="TARGET"]
    cat_cols=df.select_dtypes(include=["object"]).columns.tolist()

    num_imputer=FunctionTransformer(func=fill_num)
    cat_imputer=FunctionTransformer(func=fill_cat)

    imputer=ColumnTransformer(
        transformers=[("num",num_imputer,num_cols),
        ("cat",cat_imputer,cat_cols)]
    )

    # num_transformer=Pipeline(steps=[("num_imputer",FunctionTransformer(func=fill_num)),
    #                             ("scaler",StandardScaler())])
    # cat_transformer=Pipeline(steps=[("cat_imputer",FunctionTransformer(func=fill_cat)),
    #                              ("encoder",TargetEncoder())])

    X_tr_filled = imputer.fit_transform(X_tr)
    X_tr_filled = pd.DataFrame(X_tr_filled, columns=num_cols + cat_cols)

    num_transformer = Pipeline(steps=[
        # ("num_imputer",FunctionTransformer(func=fill_num)),
        ("scaler", StandardScaler())])
    cat_transformer = Pipeline(steps=[
        # ("cat_imputer",FunctionTransformer(func=fill_cat)),
        ("encoder", TargetEncoder())])

    preprocessor=ColumnTransformer(
    transformers=[
        ("num",num_transformer,num_cols),
        ("cat",cat_transformer,cat_cols)])
         
         #création d'une pipeline qui reprend le processeur précédent
    # avec la régression logistique
    
    
    
    pipeline_model=Pipeline(steps=[('preprocessor',preprocessor),
    ('logreg',LogisticRegression(random_state=64,
                                 solver='lbfgs',
                                 C=170.06590074301735,
                                 max_iter=2000,
                                 penalty="l2"))])

    pipeline_model.fit(X_tr_filled,y_tr)


    return pipeline_model

def train_model(df,X_tr,X_t,y_tr,y_t):
    
    # entraînement du modèle
    model=build_pipeline(df,X_tr,y_tr)
    # model.fit(X_tr,y_tr)
    
    # # prédictions
    # y_train_pred=model.predict(X_tr)
    # y_test_pred=model.predict(X_t)
    #
    # # ROC curves image
    # lrc_plot=RocCurveDisplay.from_estimator(model,X_t,y_t)
    # plt.savefig("images/results/roc_curve.jpg")
    # plt.close()
    #
    # # classification reports images
    # classification_report_image(
    #     y_tr,
    #     y_train_pred,
    #     y_t,
    #     y_test_pred)
    #
    
    # sauvegarde du modèle
    joblib.dump(model,"./models/logreg_model.joblib")




    
    
    
def main():
    logging.info("Importation des données...")
    raw_data=import_data("../application_train.csv")
    logging.info("Importation des données avec succès")
    
    # logging.info("Application de la fonction seuil_na...")
    # raw_data=seuil_na(raw_data)
    # logging.info("Application de la fonction avec succès")
    
    logging.info("Division des données...")
    train_data,Xtrain,Xtest,ytrain,ytest=data_splitting(raw_data)
    logging.info("Division des données avec succès...")
    
    # logging.info("Analyse exploratoire des données")
    # perform_eda(raw_data)
    # logging.info("Analyse exploratoire des données avec succès")
    
    logging.info("Formation du modèle")
    train_model(train_data,Xtrain,Xtest,ytrain,ytest)
    logging.info("Formation du modèle avec succès")
    
    
if __name__=="__main__":
    print("Exécution en cours")
    main()
    
    print("Fin de l'exécution avec succès")