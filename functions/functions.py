from sklearn.impute import SimpleImputer


def fill_num(df, strategy="median"):
    df_tmp = df.copy()
    imputer = SimpleImputer(strategy=strategy)
    cols = [col for col in df.columns if df[col].dtype == "float64"]
    df_tmp[cols] = imputer.fit_transform(df_tmp[cols])
    return df_tmp.values


# La fonction prend un DataFrame et un indicateur booléen most_frequent qui indique si l'on
# veut utiliser la stratégie "most_frequent" de SimpleImputer
# ou par la chaîne "Unknown"

def fill_cat(df, most_frequent=False):
    df_tmp = df.copy()

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
