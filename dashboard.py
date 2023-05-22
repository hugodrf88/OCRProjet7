from typing import Dict,Union

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from pydantic import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

import plotly.graph_objects as go
import plotly.io as pio



from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

import io
 

import joblib
#import keras
from datetime import datetime
import joblib
from datetime import date, timedelta
import seaborn as sns

#
# import plotly.express as px
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots


class PredictionVariables(BaseModel):
    params: Dict[str, Union[str, int, float]]={}  # Utilisation d'un dictionnaire pour stocker les variables
    def update_variables(self, variables_dict: Dict[str, Union[str, int, float]]):
        self.params.update(variables_dict)
dict_main_variables={'CODE_GENDER':"Sexe",#'Gender of the client'
                'DAYS_BIRTH':"Date de naissance",#"Client's age in days at the time of application"

                'NAME_EDUCATION_TYPE':"Niveau d'étude",#'Level of highest education the client achieved'
                'NAME_FAMILY_STATUS':"Status familial", #'Family status of the client'
                'NAME_HOUSING_TYPE':"Type d'habitation",#'What is the housing situation of the client (renting, living with parents, ...)'
                'FLAG_OWN_REALTY':"Possède un logement ?",#'Flag if client owns a house or flat'
                'OCCUPATION_TYPE':"Catégorie professionnelle",#'What kind of occupation does the client have'
                'DAYS_EMPLOYED':"Date d'embauche",#'How many days before the application the person started current employment'
                
                'FLAG_OWN_CAR':"Possède une voiture ?",#'Flag if client owns a car'
                # 'OWN_CAR_AGE':"Age de la voiture",#"Age of client's car"
                
                'CNT_CHILDREN':"Nombre d'enfants",#'Number of children the client has'
                'CNT_FAM_MEMBERS':"Nombre de membres de famille",# 'How many family members does client have']
                
                'AMT_INCOME_TOTAL':"Revenu annuel",#'Income of the client'
                'AMT_CREDIT':"Montant du prêt",#'Credit amount of the loan'
                'AMT_ANNUITY':"Annuité du prêt",#'Loan annuity'
                'AMT_GOODS_PRICE':"Prix du bien",#'AMT_GOODS_PRICE'
                'NAME_CONTRACT_TYPE':"Type de contrat",#''Identification if loan is cash or revolving']'

                

                }







def fill_num(df,strategy="median"):
    df_tmp=df.copy()
    imputer=SimpleImputer(strategy=strategy)
    cols=[col for col in df.columns if df[col].dtype=="float64"]
    df_tmp[cols]=imputer.fit_transform(df_tmp[cols])
    return df_tmp.values




# La fonction prend un DataFrame et un indicateur booléen most_frequent qui indique si l'on
#veut utiliser la stratégie "most_frequent" de SimpleImputer 
# ou par la chaîne "Unknown"

def fill_cat(df, most_frequent=False):
    df_tmp=df.copy()
    
    # Sélection des colonnes catégorielles dans le DataFrame
    cols = [col for col in df.columns if df[col].dtype == 'object']

    #  remplir les na dans chaque colonne catégorielle avec la valeur la plus fréquente
    if most_frequent: 
        imputer = SimpleImputer(strategy='most_frequent')
        # Appliquer la méthode fit_transform de SimpleImputer 
        # et remplacer les valeurs manquantes par la valeur la plus fréquente
        df_tmp[cols] = imputer.fit_transform(df_tmp[cols])
    # Sinon, remplir les na dans chaque colonne catégorielle avec "Unknown"
    else:
        
        df_tmp[cols] = df_tmp[cols].fillna("Unknown")

    return df_tmp.values

def preprocessor_f(df):
    num_cols=df.select_dtypes(exclude="object").columns.tolist()
    num_cols=[col for col in num_cols if col!="TARGET"]
    num_cols=[col for col in num_cols if col in dict_main_variables.keys()]

    cat_cols=df.select_dtypes(include="object").columns.tolist()
    cat_cols=[col for col in cat_cols if col in dict_main_variables.keys()]

    
    # il faut réappliquer le pipeline pour les valeurs manquantes
    # réecrivons le 
    num_transformer_fill=Pipeline(steps=[("num_imputer",FunctionTransformer(func=fill_num,kw_args={"strategy":"mean"}))])
    cat_transformer_fill=Pipeline(steps=[("cat_imputer",FunctionTransformer(func=fill_cat,kw_args={"most_frequent":False}))])
    preprocessor_fi=ColumnTransformer(transformers=[
        ("num",num_transformer_fill,num_cols),
        ("cat",cat_transformer_fill,cat_cols)
    ])
    
    preprocessor_fi.fit(df)
    return preprocessor_fi


# # Chargement de la pipeline à partir du fichier
preprocessor = joblib.load('./models/preprocessor2.joblib')
# #preprocessor_fill=joblib.load('./models/preprocessor_fill.joblib')
#
# model_logreg = joblib.load('./models/best_model.pkl')
# #model_knn=joblib.load("./models/knn_model.joblib")

# importation du dataset transformé
# data_transformed=pd.read_csv("./data/data_transformed.csv", encoding='utf-8')
# data=pd.read_csv('./data/data.csv')
target = pd.read_csv('./data/target.csv', index_col=0)
target = target.reset_index(drop=True)

st.set_option("deprecation.showPyplotGlobalUse",False)






def main():
    

    
    def load_df():
        data_sample = pd.read_csv('./data/data_sample.csv', index_col=0)
        data_sample = data_sample.reset_index(drop=True)
        return data_sample
        
    data_sample=load_df()
    
    

    
    num_cols_full=data_sample.select_dtypes(exclude=[object]).columns.to_list()
    cat_cols_full=data_sample.select_dtypes(include=[object]).columns.to_list()
    
    num_cols=data_sample.select_dtypes(exclude="object").columns.tolist()
    num_cols=[col for col in num_cols if col!="TARGET"]
    num_cols=[col for col in num_cols if col in dict_main_variables.keys()]

    cat_cols=data_sample.select_dtypes(include="object").columns.tolist()
    cat_cols=[col for col in cat_cols if col in dict_main_variables.keys()]
    
    def init_df(df):
        
        
        
        df_client=pd.DataFrame(np.nan,index=range(1),columns=df.columns)    
        # création d'un dictionnaire qui permettra de créer des variables globales à partir du nom des variables
        var_dict={}
        for col in df.columns:
            if col in num_cols_full:
                median=df[col].median()
                df_client[col]=median
        
                col=col.lower()
                var_dict[col]=median
                
            if col in cat_cols_full:
                max_value=df[col].value_counts().index[0]
                df_client[col]=max_value
                
                col=col.lower()
                var_dict[col]=max_value
        
        # générer les variables à partir du dcitionnaire
        return df_client,var_dict
    
    
    # def update_df(df):
    #     for c in main_variables:
    #         df[c]=c.globals()[c]


    #data_client,dic=init_df(data_sample)
  
    data_client=pd.read_csv("./data/data_med.csv",index_col=0)

    
    other_variables=[var for var  in data_sample.columns if var not in dict_main_variables.keys()]

    st.sidebar.header("Paramètres d'entrée")
    
    for k in dict_main_variables.keys():
        if k in cat_cols:
            vals=data_sample[k].value_counts().index.to_list()
            default_value=data_client[k].values[0]
            if len(vals)>3:
                new_value=st.sidebar.selectbox(dict_main_variables[k],vals)
                data_client[k]=data_client[k].apply(lambda x:new_value )

                
            else:
                if k=="CODE_GENDER":
                    vals=['F','M']
                
                new_value=st.sidebar.radio(dict_main_variables[k],vals)
                data_client[k]=data_client[k].apply(lambda x:new_value )
                
        else:
            if ("DAYS" in k)|("AGE" in k):
                today=date.today()
                min_date=today-timedelta(days=36500)
                new_value = st.sidebar.date_input(dict_main_variables[k], min_value=min_date,max_value=datetime.today())
                d=(date.today()-new_value).days
                data_client[k]=data_client[k].apply(lambda x:-d )
            
            else:
                new_value=st.sidebar.number_input(dict_main_variables[k],min_value=0.0,step=1.0)
                data_client[k]=data_client[k].apply(lambda x:new_value )

                
            

    
   
    
    advanced_options=st.sidebar.checkbox("Options avancées")

    

# Créer une interface utilisateur
    st.title("Prédiction de la probabilité de non-remboursement d'un prêt")

    # data_client_trans=preprocessor.transform(data_client)


    data_client_dict=data_client.to_dict(orient='records')[0]

    prediction_variables=PredictionVariables()
    prediction_variables.update_variables(data_client_dict)
  #  st.write(prediction_variables)

    response = requests.post('http://127.0.0.3:8000/prediction', json=data_client_dict)

    # Vérifier si la requête a réussi
    if response.status_code == 200:
        # Extraire les données JSON de la réponse
        df = response.json()

        # Accéder à la partie spécifique des données souhaitées
        if 'prediction' in df:
            prediction = df['prediction']
            st.markdown(f"<h2 style='font-size:36px;'> La prédiction est : {prediction}</h2>", unsafe_allow_html=True)

        else:
            st.write('Données de prédiction manquantes dans la réponse.')
    else:
        st.write('La requête a échoué avec le code de statut :', response.status_code)
    #prediction=response.json()
    #st.write(prediction)
    #rounded_prediction=str(prediction.round(2))

    st.markdown("<br><br>", unsafe_allow_html=True)  # Add line breaks
    st.markdown("<h2 style='font-size:24px;'>Résultat :</h2>", unsafe_allow_html=True)    #st.write(f"<h1 style='text-align: center; color: grey; font-size: 55px;'>{str(rounded_prediction).strip('[]')}</h1>", unsafe_allow_html=True)
    
    
    seuil=1/4
    proba=response.json()
    p=proba["prediction"]
    # Définir une variable conditionnelle
    condition = p<=seuil
    
    # Afficher un texte en vert si la condition est vraie, sinon en rouge
    if condition:
        st.write('<div style="text-align:center; font-size: 36px; font-weight: bold; color: green;">Prêt accordé ! </div>', unsafe_allow_html=True)

    else:
        st.write('<div style="text-align:center; font-size: 36px; font-weight: bold; color: red;">Prêt refusé ! </div>', unsafe_allow_html=True)
        
    st.write('<div style="text-align:center; font-size: 18px; color: grey;"><i> Seules les  probabilités inférieures à {} permettent l\'obtention du prêt</i> </div>'.format(seuil), unsafe_allow_html=True)




            

# Ajouter des options facultatives
#if checkbox:
  #  option1 = st.sidebar.slider('Option 1', min_value=0, max_value=10, value=5)
 #   option2 = st.sidebar.selectbox('Option 2', ['Option 2A', 'Option 2B', 'Option 2C'])

# Créer un bouton pour faire la prédiction


    # Ajouter les options facultatives au DataFrame
    if advanced_options:
        for k in other_variables:
            if (k in cat_cols)|("FLAG" in k):
                vals=data_sample[k].value_counts().index.to_list()
                default_value=data_client[k].values[0]
                if len(vals)>3:
                    new_value=st.sidebar.selectbox(k,vals)
                    data_client[k]=data_client[k].apply(lambda x:new_value )

                    
                else:
                    
                    
                    new_value=st.sidebar.radio(k,vals)
                    data_client[k]=data_client[k].apply(lambda x:new_value )
                    
            else:
                if ("DAYS" in k)|("AGE" in k):
                    today=date.today()
                    min_date=today-timedelta(days=36500)
                    new_value = st.sidebar.date_input(k, min_value=min_date,max_value=datetime.today())
                    d=(date.today()-new_value).days
                    data_client[k]=data_client[k].apply(lambda x:-d )
                
                else:
                    new_value=st.sidebar.number_input(k,min_value=0.0,step=1.0)
                    data_client[k]=data_client[k].apply(lambda x:new_value )
        

    # Faire la prédiction avec le modèle
   # prediction = model.predict(input_data)

    # Afficher la prédiction
 #   st.write('La prédiction est :', prediction[0])

    summary=st.sidebar.button("Afficher résumé")
    if summary:
        # data_client=update_df(data_client)
        st.write(data_client.T)
        
    # reset=st.sidebar.button("Reset")
    # if reset:
    #     # data_client=update_df(data_client)
    #     reset_my_var()iable
        
# # Define a function to reset the value of my_variable
# def reset_my_variable():
#     session_state=SessionState()
#     session_state.my_variable = "default_value"

    def knn_search(client,n=100):
        knn=KNeighborsClassifier(n_neighbors=n)
        data_trans=preprocessor.fit_transform(data_sample,target)
        knn.fit(data_trans,target)
        
        client_transformed=preprocessor.transform(client)
        distances, indices = knn.kneighbors(client_transformed, n_neighbors=n)
        clients_proches = data_sample.iloc[indices[0]]
        
        return clients_proches
    



    # comparaison=st.button("Afficher comparaisons")
    # if comparaison:
    #
    #     num_cols=data_sample.select_dtypes(exclude="object").columns.tolist()
    #     num_cols=[col for col in num_cols if col!="TARGET"]
    #     num_cols=[col for col in num_cols if col in dict_main_variables.keys()]
    #
    #
    #
    #     # cat_cols=data.select_dtypes(include="object").columns.tolist()
    #     # cat_cols=[col for col in cat_cols if col in dict_main_variables.keys()]
    #
    #     n_clients=500
    #
    #     preprocessor_fill=preprocessor_f(data_sample)
    #
    #
    #     clients_proches=knn_search(data_client,n=n_clients)
    #     index=clients_proches.index
    #
    #     preprocessor_fill.fit(data_sample,target)
    #     clients_proches_filled=preprocessor_fill.transform(clients_proches)
    #
    #     clients_proches_filled=pd.DataFrame(clients_proches_filled,columns=num_cols+cat_cols)
    #     clients_proches_filled=clients_proches_filled.set_index(index)
    #     clients_proches_filled["TARGET"]=target.iloc[index]
    #     clients_proches_filled = clients_proches_filled[list(dict_main_variables.keys()) + ["TARGET"]]
    #
    #
    #
    #
    #     for cat in cat_cols:
    #         # Data
    #         groups = clients_proches_filled[cat].value_counts().index.tolist()
    #         values_0 = [clients_proches_filled.loc[(clients_proches_filled[cat]==c)&(clients_proches_filled["TARGET"]==0),"TARGET"].count() for c in groups]
    #         values_1 = [clients_proches_filled.loc[(clients_proches_filled[cat]==c)&(clients_proches_filled["TARGET"]==1),"TARGET"].count() for c in groups]
    #
    #         fig, ax = plt.subplots()
    #
    #         # Stacked bar chart
    #         ax.bar(groups, values_0, color="green")
    #         ax.bar(groups, values_1, bottom=values_0, color="red")
    #
    #         st.markdown(f"<h3><b>{cat}</b>, <span style='font-size:24px;'>client actuel :</span></h3>", unsafe_allow_html=True)
    #         st.markdown(f"<h2><span style='color:blue'>{data_client[cat].values[0]}</span></h2>", unsafe_allow_html=True)
    #
    #
    #
    #
    #         ax.legend(["Succès", "Echec"], loc="upper right")
    #         plt.xticks(rotation=62)
    #         st.pyplot(fig)
    #
    #             # Choisir une palette de couleurs
    #     palette = sns.color_palette(["green", "red"])
    #
    #     # Créer une figure avec des sous-graphiques pour chaque variable numérique
    #     fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(50, 16))
    #
    #     # Boucle sur les variables numériques
    #     for i, num in enumerate(num_cols):
    #         # Créer le stripplot en utilisant "TARGET" comme hue
    #         sns.stripplot(x="TARGET", y=num, data=clients_proches_filled, palette=palette, ax=axes[i], size=15)
    #
    #         # Ajouter un point bleu pour la valeur 100
    #         axes[i].scatter(x=0.5, y=data_client[num], s=300, c='blue', marker='o')
    #
    #         # Ajouter un titre et des étiquettes d'axe
    #         axes[i].set_title(f"{num}", fontsize=24, fontweight="bold")
    #         axes[i].set_xlabel("TARGET", fontsize=20, fontweight="bold")
    #         axes[i].set_ylabel(num, fontsize=20, fontweight="bold")
    #         axes[i].set_xticks([0,1])
    #         axes[i].set_xticklabels(["Succès", "Échec "], fontsize=16)
    #         axes[i].tick_params(axis='both', which='major', labelsize=16)
    #         for tick in axes[i].get_xticklabels():
    #             tick.set_rotation(0)
    #
    #     # Ajuster la taille des sous-graphiques et espacer les uns des autres
    #     fig.tight_layout(pad=3)
    #
    #     # Convertir le graphique en une image pour l'afficher dans Streamlit
    #     buffer = io.BytesIO()
    #     plt.savefig(buffer, format='png')
    #     plt.close(fig)
    #     image = buffer.getvalue()
    #
    #     # Afficher l'image dans Streamlit
    #     st.image(image, use_column_width=True)
    #
    #     plt.close()
    #
    #
    #
    #     # Créer une figure avec des sous-graphiques pour chaque variable numérique
    #     fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(50, 16))
    #
    #
    #
    #     supp_cols=["RATIO_INCOME_CREDIT","RATIO_GOODS_CREDIT"]
    #
    #     clients_proches_filled["RATIO_INCOME_CREDIT"]=clients_proches_filled["AMT_INCOME_TOTAL"]/clients_proches_filled["AMT_CREDIT"]
    #     clients_proches_filled["RATIO_GOODS_CREDIT"]=clients_proches_filled["AMT_GOODS_PRICE"]/clients_proches_filled["AMT_CREDIT"]
    #
    #     data_client["RATIO_INCOME_CREDIT"]=data_client["AMT_INCOME_TOTAL"]/data_client["AMT_CREDIT"]
    #     data_client["RATIO_GOODS_CREDIT"]=data_client["AMT_GOODS_PRICE"]/data_client["AMT_CREDIT"]
    #
    #
    #
    #     # Boucle sur les variables numériques
    #     for i, supp in enumerate(supp_cols):
    #         # Créer le stripplot en utilisant "TARGET" comme hue
    #         sns.stripplot(x="TARGET", y=supp, data=clients_proches_filled, palette=palette, ax=axes[i], size=15)
    #
    #         # Ajouter un point bleu pour la valeur 100
    #         axes[i].scatter(x=0.5, y=data_client[supp], s=300, c='blue', marker='o')
    #
    #         # Ajouter un titre et des étiquettes d'axe
    #         axes[i].set_title(f"{supp}", fontsize=24, fontweight="bold")
    #         axes[i].set_xlabel("TARGET", fontsize=20, fontweight="bold")
    #         axes[i].set_ylabel(num, fontsize=20, fontweight="bold")
    #         axes[i].set_xticks([0,1])
    #         axes[i].set_xticklabels(["Succès", "Échec "], fontsize=16)
    #         axes[i].tick_params(axis='both', which='major', labelsize=16)
    #         for tick in axes[i].get_xticklabels():
    #             tick.set_rotation(0)
    #
    #     # Ajuster la taille des sous-graphiques et espacer les uns des autres
    #     fig.tight_layout(pad=3)
    #
    #     # Convertir le graphique en une image pour l'afficher dans Streamlit
    #     buffer = io.BytesIO()
    #     plt.savefig(buffer, format='png')
    #     plt.close(fig)
    #     image = buffer.getvalue()
    #
    #     # Afficher l'image dans Streamlit
    #     st.image(image, use_column_width=True)
    #
    #     plt.close()

    graphiques = st.button("Afficher graphiques")
    if graphiques :
        num_cols = data_sample.select_dtypes(exclude="object").columns.tolist()
        num_cols = [col for col in num_cols if col != "TARGET"]
        num_cols = [col for col in num_cols if col in dict_main_variables.keys()]

        # cat_cols=data.select_dtypes(include="object").columns.tolist()
        # cat_cols=[col for col in cat_cols if col in dict_main_variables.keys()]

        n_clients = 500
        st.title(f"Comparaisons avec les {n_clients} clients les plus proches :")
        preprocessor_fill = preprocessor_f(data_sample)

        clients_proches = knn_search(data_client, n=n_clients)
        index = clients_proches.index

        preprocessor_fill.fit(data_sample, target)
        clients_proches_filled = preprocessor_fill.transform(clients_proches)

        clients_proches_filled = pd.DataFrame(clients_proches_filled, columns=num_cols + cat_cols)
        clients_proches_filled = clients_proches_filled.set_index(index)
        clients_proches_filled["TARGET"] = target.iloc[index]
        clients_proches_filled = clients_proches_filled[list(dict_main_variables.keys()) + ["TARGET"]]

        for cat in cat_cols:
            groups = clients_proches_filled[cat].value_counts().index.tolist()
            values_0 = [clients_proches_filled.loc[(clients_proches_filled[cat] == c) & (
                        clients_proches_filled["TARGET"] == 0), "TARGET"].count() for c in groups]
            values_1 = [clients_proches_filled.loc[(clients_proches_filled[cat] == c) & (
                        clients_proches_filled["TARGET"] == 1), "TARGET"].count() for c in groups]

            fig = go.Figure()

            # Stacked bar chart
            fig.add_trace(go.Bar(x=groups, y=values_0, name='Succès', marker_color='green'))
            fig.add_trace(go.Bar(x=groups, y=values_1, name='Echec', marker_color='red'))

            fig.update_layout(
                title=f"{cat}, client actuel: {data_client[cat].values[0]}",
                xaxis=dict(title=cat),
                yaxis=dict(title='Nombre de clients'),
                barmode='stack'
            )

            # Afficher le graphique interactif
            st.plotly_chart(fig)

        # Choisir une palette de couleurs
        palette = ["green", "red"]

        # Créer un DataFrame pour les variables numériques
        num_data = pd.DataFrame(clients_proches_filled[num_cols])
        num_data["TARGET"] = clients_proches_filled["TARGET"]

        # Boucle sur les variables numériques
        for num in num_cols:
            # Créer un stripplot interactif avec Plotly
            fig = go.Figure()

            # Ajouter les points pour les valeurs de succès et d'échec
            fig.add_trace(
                go.Box(x=num_data[num][num_data["TARGET"] == 0], name="Succès", marker=dict(color="green")))

            # Ajouter un scatterplot pour la valeur du client actuel
            fig.add_trace(go.Box(x=data_client[num], name="Client actuel", marker=dict(color="blue")))

            # Ajouter les points pour les valeurs de succès et d'échec
            fig.add_trace(go.Box(x=num_data[num][num_data["TARGET"] == 1], name="Échec", marker=dict(color="red")))



            # Modifier l'apparence du graphique
            fig.update_layout(
                title=f"{num}",
                xaxis=dict(title="TARGET", ticktext=["Succès", "Échec"]),
                yaxis=dict(title=num),
            )

            # Afficher le graphique interactif dans Streamlit
            st.plotly_chart(fig)

        supp_cols = ["RATIO_INCOME_CREDIT", "RATIO_GOODS_CREDIT"]

        clients_proches_filled["RATIO_INCOME_CREDIT"] = clients_proches_filled["AMT_INCOME_TOTAL"] / \
                                                        clients_proches_filled["AMT_CREDIT"]
        clients_proches_filled["RATIO_GOODS_CREDIT"] = clients_proches_filled["AMT_GOODS_PRICE"] / \
                                                       clients_proches_filled["AMT_CREDIT"]

        data_client["RATIO_INCOME_CREDIT"] = data_client["AMT_INCOME_TOTAL"] / data_client["AMT_CREDIT"]
        data_client["RATIO_GOODS_CREDIT"] = data_client["AMT_GOODS_PRICE"] / data_client["AMT_CREDIT"]

        # Créer un DataFrame pour les variables supplémentaires
        supp_data = pd.DataFrame(clients_proches_filled[supp_cols])
        supp_data["TARGET"] = clients_proches_filled["TARGET"]

        # Boucle sur les variables supplémentaires
        for i, supp in enumerate(supp_cols):
            # Créer un stripplot interactif avec Plotly
            fig = go.Figure()

            # Ajouter les points pour les valeurs de succès et d'échec
            fig.add_trace(go.Box(x=supp_data[supp][supp_data["TARGET"] == 0], name="Succès",marker=dict(color="green")))

            # Ajouter un scatterplot pour la valeur du client actuel
            fig.add_trace(go.Box(x=data_client[supp],name="Client actuel",marker=dict(color="blue")))

            # Ajouter les points pour les valeurs de succès et d'échec
            fig.add_trace(go.Box(x=supp_data[supp][supp_data["TARGET"] == 1], name="Échec",marker=dict(color="red")))



            # Modifier l'apparence du graphique
            fig.update_layout(
                title=f"{supp}",
                xaxis=dict(title="TARGET", ticktext=["Succès", "Échec"]),
                yaxis=dict(title=supp),
            )

            # Afficher le graphique interactif dans Streamlit
            st.plotly_chart(fig)



    
if __name__=='__main__':
    main()
    
    

            