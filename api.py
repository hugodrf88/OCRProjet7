
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import logging
import dill as pickle
from functions.functions import fill_cat,fill_num


logreg_model = joblib.load('./models/logreg_model.joblib')


logging.basicConfig(level=logging.DEBUG)





app = FastAPI()


class PredictionRequest(BaseModel):
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_TYPE_SUITE: str
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    OWN_CAR_AGE: float
    FLAG_MOBIL: int
    FLAG_EMP_PHONE: int
    FLAG_WORK_PHONE: int
    FLAG_CONT_MOBILE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    WEEKDAY_APPR_PROCESS_START: str
    HOUR_APPR_PROCESS_START: int
    REG_REGION_NOT_LIVE_REGION: int
    REG_REGION_NOT_WORK_REGION: int
    LIVE_REGION_NOT_WORK_REGION: int
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    LIVE_CITY_NOT_WORK_CITY: int
    ORGANIZATION_TYPE: str
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    APARTMENTS_AVG: float
    BASEMENTAREA_AVG: float
    YEARS_BEGINEXPLUATATION_AVG: float
    YEARS_BUILD_AVG: float
    COMMONAREA_AVG: float
    ELEVATORS_AVG: float
    ENTRANCES_AVG: float
    FLOORSMAX_AVG: float
    FLOORSMIN_AVG: float
    LANDAREA_AVG: float
    LIVINGAPARTMENTS_AVG: float
    LIVINGAREA_AVG: float
    NONLIVINGAPARTMENTS_AVG: float
    NONLIVINGAREA_AVG: float
    APARTMENTS_MODE: float
    BASEMENTAREA_MODE: float
    YEARS_BEGINEXPLUATATION_MODE: float
    YEARS_BUILD_MODE: float
    COMMONAREA_MODE: float
    ELEVATORS_MODE: float
    ENTRANCES_MODE: float
    FLOORSMAX_MODE: float
    FLOORSMIN_MODE: float
    LANDAREA_MODE: float
    LIVINGAPARTMENTS_MODE: float
    LIVINGAREA_MODE: float
    NONLIVINGAPARTMENTS_MODE: float
    NONLIVINGAREA_MODE: float
    APARTMENTS_MEDI: float
    BASEMENTAREA_MEDI: float
    YEARS_BEGINEXPLUATATION_MEDI: float
    YEARS_BUILD_MEDI: float
    COMMONAREA_MEDI: float
    ELEVATORS_MEDI: float
    ENTRANCES_MEDI: float
    FLOORSMAX_MEDI: float
    FLOORSMIN_MEDI: float
    LANDAREA_MEDI: float
    LIVINGAPARTMENTS_MEDI: float
    LIVINGAREA_MEDI: float
    NONLIVINGAPARTMENTS_MEDI: float
    NONLIVINGAREA_MEDI: float
    FONDKAPREMONT_MODE: str
    HOUSETYPE_MODE: str
    TOTALAREA_MODE: float
    WALLSMATERIAL_MODE: str
    EMERGENCYSTATE_MODE: str
    OBS_30_CNT_SOCIAL_CIRCLE: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_2: int
    FLAG_DOCUMENT_3: int
    FLAG_DOCUMENT_4: int
    FLAG_DOCUMENT_5: int
    FLAG_DOCUMENT_6: int
    FLAG_DOCUMENT_7: int
    FLAG_DOCUMENT_8: int
    FLAG_DOCUMENT_9: int
    FLAG_DOCUMENT_10: int
    FLAG_DOCUMENT_11: int
    FLAG_DOCUMENT_12: int
    FLAG_DOCUMENT_13: int
    FLAG_DOCUMENT_14: int
    FLAG_DOCUMENT_15: int
    FLAG_DOCUMENT_16: int
    FLAG_DOCUMENT_17: int
    FLAG_DOCUMENT_18: int
    FLAG_DOCUMENT_19: int
    FLAG_DOCUMENT_20: int
    FLAG_DOCUMENT_21: int
    AMT_REQ_CREDIT_BUREAU_HOUR: float
    AMT_REQ_CREDIT_BUREAU_DAY: float
    AMT_REQ_CREDIT_BUREAU_WEEK: float
    AMT_REQ_CREDIT_BUREAU_MON: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float


class PredictionResponse(BaseModel):
    prediction: float



@app.get("/")
def read_root():
    print("Hello, World!")
    return {"message": "Hello, World!"}

@app.post('/prediction', response_model=PredictionResponse)
def prediction(request: PredictionRequest):
    logging.basicConfig(level=logging.DEBUG)

    logging.debug("Votre message de débogage ici")
    dict_values = {
        "NAME_CONTRACT_TYPE": request.NAME_CONTRACT_TYPE,
        "CODE_GENDER": request.CODE_GENDER,
        "FLAG_OWN_CAR": request.FLAG_OWN_CAR,
        "FLAG_OWN_REALTY": request.FLAG_OWN_REALTY,
        "CNT_CHILDREN": request.CNT_CHILDREN,
        "AMT_INCOME_TOTAL": request.AMT_INCOME_TOTAL,
        "AMT_CREDIT": request.AMT_CREDIT,
        "AMT_ANNUITY": request.AMT_ANNUITY,
        "AMT_GOODS_PRICE": request.AMT_GOODS_PRICE,
        "NAME_TYPE_SUITE": request.NAME_TYPE_SUITE,
        "NAME_INCOME_TYPE": request.NAME_INCOME_TYPE,
        "NAME_EDUCATION_TYPE": request.NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS": request.NAME_FAMILY_STATUS,
        "NAME_HOUSING_TYPE": request.NAME_HOUSING_TYPE,
        "REGION_POPULATION_RELATIVE": request.REGION_POPULATION_RELATIVE,
        "DAYS_BIRTH": request.DAYS_BIRTH,
        "DAYS_EMPLOYED": request.DAYS_EMPLOYED,
        "DAYS_REGISTRATION": request.DAYS_REGISTRATION,
        "DAYS_ID_PUBLISH": request.DAYS_ID_PUBLISH,
        "OWN_CAR_AGE": request.OWN_CAR_AGE,
        "FLAG_MOBIL": request.FLAG_MOBIL,
        "FLAG_EMP_PHONE": request.FLAG_EMP_PHONE,
        "FLAG_WORK_PHONE": request.FLAG_WORK_PHONE,
        "FLAG_CONT_MOBILE": request.FLAG_CONT_MOBILE,
        "FLAG_PHONE": request.FLAG_PHONE,
        "FLAG_EMAIL": request.FLAG_EMAIL,
        "OCCUPATION_TYPE": request.OCCUPATION_TYPE,
        "CNT_FAM_MEMBERS": request.CNT_FAM_MEMBERS,
        "REGION_RATING_CLIENT": request.REGION_RATING_CLIENT,
        "REGION_RATING_CLIENT_W_CITY": request.REGION_RATING_CLIENT_W_CITY,
        "WEEKDAY_APPR_PROCESS_START": request.WEEKDAY_APPR_PROCESS_START,
        "HOUR_APPR_PROCESS_START": request.HOUR_APPR_PROCESS_START,
        "REG_REGION_NOT_LIVE_REGION": request.REG_REGION_NOT_LIVE_REGION,
        "REG_REGION_NOT_WORK_REGION": request.REG_REGION_NOT_WORK_REGION,
        "LIVE_REGION_NOT_WORK_REGION": request.LIVE_REGION_NOT_WORK_REGION,
        "REG_CITY_NOT_LIVE_CITY": request.REG_CITY_NOT_LIVE_CITY,
        "REG_CITY_NOT_WORK_CITY": request.REG_CITY_NOT_WORK_CITY,
        "LIVE_CITY_NOT_WORK_CITY": request.LIVE_CITY_NOT_WORK_CITY,
        "ORGANIZATION_TYPE": request.ORGANIZATION_TYPE,
        "EXT_SOURCE_1": request.EXT_SOURCE_1,
        "EXT_SOURCE_2": request.EXT_SOURCE_2,
        "EXT_SOURCE_3": request.EXT_SOURCE_3,
        "APARTMENTS_AVG": request.APARTMENTS_AVG,
        "BASEMENTAREA_AVG": request.BASEMENTAREA_AVG,
        "YEARS_BEGINEXPLUATATION_AVG": request.YEARS_BEGINEXPLUATATION_AVG,
        "YEARS_BUILD_AVG": request.YEARS_BUILD_AVG,
        "COMMONAREA_AVG": request.COMMONAREA_AVG,
        "ELEVATORS_AVG": request.ELEVATORS_AVG,
        "ENTRANCES_AVG": request.ENTRANCES_AVG,
        "FLOORSMAX_AVG": request.FLOORSMAX_AVG,
        "FLOORSMIN_AVG": request.FLOORSMIN_AVG,
        "LANDAREA_AVG": request.LANDAREA_AVG,
        "LIVINGAPARTMENTS_AVG": request.LIVINGAPARTMENTS_AVG,
        "LIVINGAREA_AVG": request.LIVINGAREA_AVG,
        "NONLIVINGAPARTMENTS_AVG": request.NONLIVINGAPARTMENTS_AVG,
        "NONLIVINGAREA_AVG": request.NONLIVINGAREA_AVG,
        "APARTMENTS_MODE": request.APARTMENTS_MODE,
        "BASEMENTAREA_MODE": request.BASEMENTAREA_MODE,
        "YEARS_BEGINEXPLUATATION_MODE": request.YEARS_BEGINEXPLUATATION_MODE,
        "YEARS_BUILD_MODE": request.YEARS_BUILD_MODE,
        "COMMONAREA_MODE": request.COMMONAREA_MODE,
        "ELEVATORS_MODE": request.ELEVATORS_MODE,
        "ENTRANCES_MODE": request.ENTRANCES_MODE,
        "FLOORSMAX_MODE": request.FLOORSMAX_MODE,
        "FLOORSMIN_MODE": request.FLOORSMIN_MODE,
        "LANDAREA_MODE": request.LANDAREA_MODE,
        "LIVINGAPARTMENTS_MODE": request.LIVINGAPARTMENTS_MODE,
        "LIVINGAREA_MODE": request.LIVINGAREA_MODE,
        "NONLIVINGAPARTMENTS_MODE": request.NONLIVINGAPARTMENTS_MODE,
        "NONLIVINGAREA_MODE": request.NONLIVINGAREA_MODE,
        "APARTMENTS_MEDI": request.APARTMENTS_MEDI,
        "BASEMENTAREA_MEDI": request.BASEMENTAREA_MEDI,
        "YEARS_BEGINEXPLUATATION_MEDI": request.YEARS_BEGINEXPLUATATION_MEDI,
        "YEARS_BUILD_MEDI": request.YEARS_BUILD_MEDI,
        "COMMONAREA_MEDI": request.COMMONAREA_MEDI,
        "ELEVATORS_MEDI": request.ELEVATORS_MEDI,
        "ENTRANCES_MEDI": request.ENTRANCES_MEDI,
        "FLOORSMAX_MEDI": request.FLOORSMAX_MEDI,
        "FLOORSMIN_MEDI": request.FLOORSMIN_MEDI,
        "LANDAREA_MEDI": request.LANDAREA_MEDI,
        "LIVINGAPARTMENTS_MEDI": request.LIVINGAPARTMENTS_MEDI,
        "LIVINGAREA_MEDI": request.LIVINGAREA_MEDI,
        "NONLIVINGAPARTMENTS_MEDI": request.NONLIVINGAPARTMENTS_MEDI,
        "NONLIVINGAREA_MEDI": request.NONLIVINGAREA_MEDI,
        "FONDKAPREMONT_MODE": request.FONDKAPREMONT_MODE,
        "HOUSETYPE_MODE": request.HOUSETYPE_MODE,
        "TOTALAREA_MODE": request.TOTALAREA_MODE,
        "WALLSMATERIAL_MODE": request.WALLSMATERIAL_MODE,
        "EMERGENCYSTATE_MODE": request.EMERGENCYSTATE_MODE,
        "OBS_30_CNT_SOCIAL_CIRCLE": request.OBS_30_CNT_SOCIAL_CIRCLE,
        "DEF_30_CNT_SOCIAL_CIRCLE": request.DEF_30_CNT_SOCIAL_CIRCLE,
        "OBS_60_CNT_SOCIAL_CIRCLE": request.OBS_60_CNT_SOCIAL_CIRCLE,
        "DEF_60_CNT_SOCIAL_CIRCLE": request.DEF_60_CNT_SOCIAL_CIRCLE,
        "DAYS_LAST_PHONE_CHANGE": request.DAYS_LAST_PHONE_CHANGE,
        "FLAG_DOCUMENT_2": request.FLAG_DOCUMENT_2,
        "FLAG_DOCUMENT_3": request.FLAG_DOCUMENT_3,
        "FLAG_DOCUMENT_4": request.FLAG_DOCUMENT_4,
        "FLAG_DOCUMENT_5": request.FLAG_DOCUMENT_5,
        "FLAG_DOCUMENT_6": request.FLAG_DOCUMENT_6,
        "FLAG_DOCUMENT_7": request.FLAG_DOCUMENT_7,
        "FLAG_DOCUMENT_8": request.FLAG_DOCUMENT_8,
        "FLAG_DOCUMENT_9": request.FLAG_DOCUMENT_9,
        "FLAG_DOCUMENT_10": request.FLAG_DOCUMENT_10,
        "FLAG_DOCUMENT_11": request.FLAG_DOCUMENT_11,
        "FLAG_DOCUMENT_12": request.FLAG_DOCUMENT_12,
        "FLAG_DOCUMENT_13": request.FLAG_DOCUMENT_13,
        "FLAG_DOCUMENT_14": request.FLAG_DOCUMENT_14,
        "FLAG_DOCUMENT_15": request.FLAG_DOCUMENT_15,
        "FLAG_DOCUMENT_16": request.FLAG_DOCUMENT_16,
        "FLAG_DOCUMENT_17": request.FLAG_DOCUMENT_17,
        "FLAG_DOCUMENT_18": request.FLAG_DOCUMENT_18,
        "FLAG_DOCUMENT_19": request.FLAG_DOCUMENT_19,
        "FLAG_DOCUMENT_20": request.FLAG_DOCUMENT_20,
        "FLAG_DOCUMENT_21": request.FLAG_DOCUMENT_21,
        "AMT_REQ_CREDIT_BUREAU_HOUR": request.AMT_REQ_CREDIT_BUREAU_HOUR,
        "AMT_REQ_CREDIT_BUREAU_DAY": request.AMT_REQ_CREDIT_BUREAU_DAY,
        "AMT_REQ_CREDIT_BUREAU_WEEK": request.AMT_REQ_CREDIT_BUREAU_WEEK,
        "AMT_REQ_CREDIT_BUREAU_MON": request.AMT_REQ_CREDIT_BUREAU_MON,
        "AMT_REQ_CREDIT_BUREAU_QRT": request.AMT_REQ_CREDIT_BUREAU_QRT,
        "AMT_REQ_CREDIT_BUREAU_YEAR": request.AMT_REQ_CREDIT_BUREAU_YEAR,
    }



    df_values=pd.DataFrame(dict_values,index=[0])
    logging.debug("Taille du df"+str(df_values.shape[0]))
    logging.debug("Nombre de variables du df"+str(df_values.shape[1]))

    prediction_proba=logreg_model.predict_proba(df_values)
    prediction_proba=prediction_proba[0][1]
    prediction_proba=prediction_proba.round(2)
    logging.debug(prediction_proba)


    return {"prediction":prediction_proba}


if __name__ == '__main__':


    uvicorn.run(app, host='127.0.0.3', port=8000)