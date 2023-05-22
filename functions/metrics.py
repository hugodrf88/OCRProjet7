from sklearn.metrics import confusion_matrix


# Définition de la fonction qui calcule le score métier personnalisé
def custom_scorer(y_true, y_pred):
    # Calcul de la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calcul du score métier personnalisé
    score = (-10 * fn - fp + tn + 10 * tp) / (tn + tp + fn + fp)

    # Retourne le score métier personnalisé
    return score