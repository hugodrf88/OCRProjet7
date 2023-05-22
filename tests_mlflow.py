from functions.metrics import custom_scorer
from sklearn.metrics import make_scorer

# importation du score personnalisé
custom_score=make_scorer(custom_scorer,greater_is_better=True)