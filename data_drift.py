import pandas as pd

from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
import joblib




loaded_model=joblib.load("./models/best_model.pkl")

reference_df=pd.read_csv("./data/application_train.csv")
production_df=pd.read_csv("./data/application_test.csv")
reference_df=reference_df.drop("SK_ID_CURR",axis=1)
production_df=production_df.drop("SK_ID_CURR",axis=1)
y_true=reference_df["TARGET"]
reference_df=reference_df.drop("TARGET",axis=1)

# numerical_features = ["bedrooms", "condition"]
# column_mapping = ColumnMapping()

# column_mapping.datetime = "date"
# column_mapping.numerical_features = numerical_features

data_drift_dashboard = Dashboard(tabs=[DataDriftTab(verbose_level=0)])

reference_pred=loaded_model.predict(reference_df)
production_pred=loaded_model.predict(production_df)

data_drift_dashboard.calculate(reference_df, production_df, 
                               #column_mapping=column_mapping
                               )
data_drift_dashboard.show(mode="inline")


data_drift_dashboard.save("data_drift.html")
