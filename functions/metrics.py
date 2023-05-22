from sklearn.metrics import confusion_matrix


# Définition de la fonction qui calcule le score métier personnalisé
def custom_scorer(y_true, y_pred):
    """
    Calcule un score métier personnalisé en utilisant la matrice de confusion.

    Args:
        y_true (array-like): Les vraies valeurs de la variable cible.
        y_pred (array-like): Les valeurs prédites de la variable cible.

    Returns:
        float: Le score métier personnalisé.
    """

    # Calcul de la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calcul du score métier personnalisé
    score = (-10 * fn - fp + tn + 10 * tp) / (tn + tp + fn + fp)

    # Retourne le score métier personnalisé
    return score
