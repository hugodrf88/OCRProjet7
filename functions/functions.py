from sklearn.impute import SimpleImputer


def fill_num(df, strategy="median"):
    """
    Remplit les valeurs manquantes des variables numériques d'un dataframe avec une stratégie donnée.

    Args:
        df (pandas.DataFrame): Le dataframe contenant les variables numériques avec des valeurs manquantes.
        strategy (str, optional): La stratégie de remplissage à utiliser. Par défaut, "median".

    Returns:
        numpy.ndarray: Un tableau numpy contenant le dataframe avec les valeurs manquantes remplacées.
    """

    # Copie le dataframe pour éviter de modifier l'original
    df_tmp = df.copy()

    # Initialise un objet SimpleImputer avec la stratégie de remplissage spécifiée
    imputer = SimpleImputer(strategy=strategy)

    # Sélectionne les colonnes numériques du dataframe
    cols = [col for col in df.columns if df[col].dtype == "float64"]

    # Remplit les valeurs manquantes des colonnes sélectionnées en utilisant l'objet SimpleImputer
    df_tmp[cols] = imputer.fit_transform(df_tmp[cols])

    # Renvoie le dataframe modifié sous forme d'un tableau numpy
    return df_tmp.values





def fill_cat(df, most_frequent=False):
    """
    Remplit les valeurs manquantes des variables catégorielles d'un dataframe avec la valeur la plus fréquente ou "Unknown".

    Args:
        df (pandas.DataFrame): Le dataframe contenant les variables catégorielles avec des valeurs manquantes.
        most_frequent (bool, optional): Indique si la valeur la plus fréquente doit être utilisée pour le remplissage.
            Par défaut, False (remplissage avec "Unknown").

    Returns:
        numpy.ndarray: Un tableau numpy contenant le dataframe avec les valeurs manquantes remplacées.
    """

    # Copie le dataframe pour éviter de modifier l'original
    df_tmp = df.copy()

    # Sélectionne les colonnes catégorielles du dataframe
    cols = [col for col in df.columns if df[col].dtype == 'object']

    # Remplit les valeurs manquantes dans chaque colonne catégorielle
    if most_frequent:
        # Si most_frequent est True, utilise la valeur la plus fréquente pour le remplissage
        imputer = SimpleImputer(strategy='most_frequent')
        df_tmp[cols] = imputer.fit_transform(df_tmp[cols])
    else:
        # Sinon, remplace les valeurs manquantes par "Unknown"
        df_tmp[cols] = df_tmp[cols].fillna("Unknown")

    # Renvoie le dataframe modifié sous forme d'un tableau numpy
    return df_tmp.values
