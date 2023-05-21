
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import logging
import dill as pickle


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'fill_num':
            from functions.functions import fill_num
            return fill_num

        if name == 'fill_cat':
            from functions.functions import fill_cat
            return fill_cat
        return super().find_class(module, name)

current_model = 'logreg_model.pkl'

model_logreg = CustomUnpickler(open('models/' + current_model, 'rb')).load()

logging.basicConfig(level=logging.DEBUG)





app = FastAPI()


class PredictionRequest(BaseModel):
    name_contract_type: str
    code_gender: str
    flag_own_car: str
    flag_own_realty: str
    cnt_children: int
    amt_income_total: float
    amt_credit: float
    amt_annuity: float
    amt_goods_price: float
    name_type_suite: str
    name_income_type: str
    name_education_type: str
    name_family_status: str
    name_housing_type: str
    region_population_relative: float
    days_birth: int
    days_employed: int
    days_registration: float
    days_id_publish: int
    own_car_age: float
    flag_mobil: int
    flag_emp_phone: int
    flag_work_phone: int
    flag_cont_mobile: int
    flag_phone: int
    flag_email: int
    occupation_type: str
    cnt_fam_members: float
    region_rating_client: int
    region_rating_client_w_city: int
    weekday_appr_process_start: str
    hour_appr_process_start: int
    reg_region_not_live_region: int
    reg_region_not_work_region: int
    live_region_not_work_region: int
    reg_city_not_live_city: int
    reg_city_not_work_city: int
    live_city_not_work_city: int
    organization_type: str
    ext_source_1: float
    ext_source_2: float
    ext_source_3: float
    apartments_avg: float
    basementarea_avg: float
    years_beginexpluatation_avg: float
    years_build_avg: float
    commonarea_avg: float
    elevators_avg: float
    entrances_avg: float
    floorsmax_avg: float
    floorsmin_avg: float
    landarea_avg: float
    livingapartments_avg: float
    livingarea_avg: float
    nonlivingapartments_avg: float
    nonlivingarea_avg: float
    apartments_mode: float
    basementarea_mode: float
    years_beginexpluatation_mode: float
    years_build_mode: float
    commonarea_mode: float
    elevators_mode: float
    entrances_mode: float
    floorsmax_mode: float
    floorsmin_mode: float
    landarea_mode: float
    livingapartments_mode: float
    livingarea_mode: float
    nonlivingapartments_mode: float
    nonlivingarea_mode: float
    apartments_medi: float
    basementarea_medi: float
    years_beginexpluatation_medi: float
    years_build_medi: float
    commonarea_medi: float
    elevators_medi: float
    entrances_medi: float
    floorsmax_medi: float
    floorsmin_medi: float
    landarea_medi: float
    livingapartments_medi: float
    livingarea_medi: float
    nonlivingapartments_medi: float
    nonlivingarea_medi: float
    fondkapremont_mode: str
    housetype_mode: str
    totalarea_mode: float
    wallsmaterial_mode: str
    emergencystate_mode: str
    obs_30_cnt_social_circle: float
    def_30_cnt_social_circle: float
    obs_60_cnt_social_circle: float
    def_60_cnt_social_circle: float
    days_last_phone_change: float
    flag_document_2: int
    flag_document_3: int
    flag_document_4: int
    flag_document_5: int
    flag_document_6: int
    flag_document_7: int
    flag_document_8: int
    flag_document_9: int
    flag_document_10: int
    flag_document_11: int
    flag_document_12: int
    flag_document_13: int
    flag_document_14: int
    flag_document_15: int
    flag_document_16: int
    flag_document_17: int
    flag_document_18: int
    flag_document_19: int
    flag_document_20: int
    flag_document_21: int
    amt_req_credit_bureau_hour: float
    amt_req_credit_bureau_day: float
    amt_req_credit_bureau_week: float
    amt_req_credit_bureau_mon: float
    amt_req_credit_bureau_qrt: float
    amt_req_credit_bureau_year: float


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
    "NAME_CONTRACT_TYPE": request.name_contract_type,
    "CODE_GENDER": request.code_gender,
    "FLAG_OWN_CAR": request.flag_own_car,
    "FLAG_OWN_REALTY": request.flag_own_realty,
    "CNT_CHILDREN": request.cnt_children,
    "AMT_INCOME_TOTAL": request.amt_income_total,
    "AMT_CREDIT": request.amt_credit,
    "AMT_ANNUITY": request.amt_annuity,
    "AMT_GOODS_PRICE": request.amt_goods_price,
    "NAME_TYPE_SUITE": request.name_type_suite,
    "NAME_INCOME_TYPE": request.name_income_type,
    "NAME_EDUCATION_TYPE": request.name_education_type,
    "NAME_FAMILY_STATUS": request.name_family_status,
    "NAME_HOUSING_TYPE": request.name_housing_type,
    "REGION_POPULATION_RELATIVE": request.region_population_relative,
    "DAYS_BIRTH": request.days_birth,
    "DAYS_EMPLOYED": request.days_employed,
    "DAYS_REGISTRATION": request.days_registration,
    "DAYS_ID_PUBLISH": request.days_id_publish,
    "OWN_CAR_AGE": request.own_car_age,
    "FLAG_MOBIL": request.flag_mobil,
    "FLAG_EMP_PHONE": request.flag_emp_phone,
    "FLAG_WORK_PHONE": request.flag_work_phone,
    "FLAG_CONT_MOBILE": request.flag_cont_mobile,
    "FLAG_PHONE": request.flag_phone,
    "FLAG_EMAIL": request.flag_email,
    "OCCUPATION_TYPE": request.occupation_type,
    "CNT_FAM_MEMBERS": request.cnt_fam_members,
    "REGION_RATING_CLIENT": request.region_rating_client,
    "REGION_RATING_CLIENT_W_CITY": request.region_rating_client_w_city,
    "WEEKDAY_APPR_PROCESS_START": request.weekday_appr_process_start,
    "HOUR_APPR_PROCESS_START": request.hour_appr_process_start,
    "REG_REGION_NOT_LIVE_REGION": request.reg_region_not_live_region,
    "REG_REGION_NOT_WORK_REGION": request.reg_region_not_work_region,
    "LIVE_REGION_NOT_WORK_REGION": request.live_region_not_work_region,
    "REG_CITY_NOT_LIVE_CITY": request.reg_city_not_live_city,
    "REG_CITY_NOT_WORK_CITY": request.reg_city_not_work_city,
    "LIVE_CITY_NOT_WORK_CITY": request.live_city_not_work_city,
    "ORGANIZATION_TYPE": request.organization_type,
    "EXT_SOURCE_1": request.ext_source_1,
    "EXT_SOURCE_2": request.ext_source_2,
    "EXT_SOURCE_3": request.ext_source_3,
    "APARTMENTS_AVG": request.apartments_avg,
    "BASEMENTAREA_AVG": request.basementarea_avg,
    "YEARS_BEGINEXPLUATATION_AVG": request.years_beginexpluatation_avg,
    "YEARS_BUILD_AVG": request.years_build_avg,
    "COMMONAREA_AVG": request.commonarea_avg,
    "ELEVATORS_AVG": request.elevators_avg,
    "ENTRANCES_AVG": request.entrances_avg,
    "FLOORSMAX_AVG": request.floorsmax_avg,
    "FLOORSMIN_AVG": request.floorsmin_avg,
    "LANDAREA_AVG": request.landarea_avg,
    "LIVINGAPARTMENTS_AVG": request.livingapartments_avg,
    "LIVINGAREA_AVG": request.livingarea_avg,
    "NONLIVINGAPARTMENTS_AVG": request.nonlivingapartments_avg,
    "NONLIVINGAREA_AVG": request.nonlivingarea_avg,
    "APARTMENTS_MODE": request.apartments_mode,
    "BASEMENTAREA_MODE": request.basementarea_mode,
    "YEARS_BEGINEXPLUATATION_MODE": request.years_beginexpluatation_mode,
    "YEARS_BUILD_MODE": request.years_build_mode,
    "COMMONAREA_MODE": request.commonarea_mode,
    "ELEVATORS_MODE": request.elevators_mode,
    "ENTRANCES_MODE": request.entrances_mode,
    "FLOORSMAX_MODE": request.floorsmax_mode,
    "FLOORSMIN_MODE": request.floorsmin_mode,
    "LANDAREA_MODE": request.landarea_mode,
    "LIVINGAPARTMENTS_MODE": request.livingapartments_mode,
    "LIVINGAREA_MODE": request.livingarea_mode,
    "NONLIVINGAPARTMENTS_MODE": request.nonlivingapartments_mode,
    "NONLIVINGAREA_MODE": request.nonlivingarea_mode,
    "APARTMENTS_MEDI": request.apartments_medi,
    "BASEMENTAREA_MEDI": request.basementarea_medi,
    "YEARS_BEGINEXPLUATATION_MEDI": request.years_beginexpluatation_medi,
    "YEARS_BUILD_MEDI": request.years_build_medi,
    "COMMONAREA_MEDI": request.commonarea_medi,
    "ELEVATORS_MEDI": request.elevators_medi,
    "ENTRANCES_MEDI": request.entrances_medi,
    "FLOORSMAX_MEDI": request.floorsmax_medi,
    "FLOORSMIN_MEDI": request.floorsmin_medi,
    "LANDAREA_MEDI": request.landarea_medi,
    "LIVINGAPARTMENTS_MEDI": request.livingapartments_medi,
    "LIVINGAREA_MEDI": request.livingarea_medi,
    "NONLIVINGAPARTMENTS_MEDI": request.nonlivingapartments_medi,
    "NONLIVINGAREA_MEDI": request.nonlivingarea_medi,
    "FONDKAPREMONT_MODE": request.fondkapremont_mode,
    "HOUSETYPE_MODE": request.housetype_mode,
    "TOTALAREA_MODE": request.totalarea_mode,
    "WALLSMATERIAL_MODE": request.wallsmaterial_mode,
    "EMERGENCYSTATE_MODE": request.emergencystate_mode,
    "OBS_30_CNT_SOCIAL_CIRCLE": request.obs_30_cnt_social_circle,
    "DEF_30_CNT_SOCIAL_CIRCLE": request.def_30_cnt_social_circle,
    "OBS_60_CNT_SOCIAL_CIRCLE": request.obs_60_cnt_social_circle,
    "DEF_60_CNT_SOCIAL_CIRCLE": request.def_60_cnt_social_circle,
    "DAYS_LAST_PHONE_CHANGE": request.days_last_phone_change,
    "FLAG_DOCUMENT_2": request.flag_document_2,
    "FLAG_DOCUMENT_3": request.flag_document_3,
    "FLAG_DOCUMENT_4": request.flag_document_4,
    "FLAG_DOCUMENT_5": request.flag_document_5,
    "FLAG_DOCUMENT_6": request.flag_document_6,
    "FLAG_DOCUMENT_7": request.flag_document_7,
    "FLAG_DOCUMENT_8": request.flag_document_8,
    "FLAG_DOCUMENT_9": request.flag_document_9,
    "FLAG_DOCUMENT_10": request.flag_document_10,
    "FLAG_DOCUMENT_11": request.flag_document_11,
    "FLAG_DOCUMENT_12": request.flag_document_12,
    "FLAG_DOCUMENT_13": request.flag_document_13,
    "FLAG_DOCUMENT_14": request.flag_document_14,
    "FLAG_DOCUMENT_15": request.flag_document_15,
    "FLAG_DOCUMENT_16": request.flag_document_16,
    "FLAG_DOCUMENT_17": request.flag_document_17,
    "FLAG_DOCUMENT_18": request.flag_document_18,
    "FLAG_DOCUMENT_19": request.flag_document_19,
    "FLAG_DOCUMENT_20": request.flag_document_20,
    "FLAG_DOCUMENT_21": request.flag_document_21,
    "AMT_REQ_CREDIT_BUREAU_HOUR": request.amt_req_credit_bureau_hour,
    "AMT_REQ_CREDIT_BUREAU_DAY": request.amt_req_credit_bureau_day,
    "AMT_REQ_CREDIT_BUREAU_WEEK": request.amt_req_credit_bureau_week,
    "AMT_REQ_CREDIT_BUREAU_MON": request.amt_req_credit_bureau_mon,
    "AMT_REQ_CREDIT_BUREAU_QRT": request.amt_req_credit_bureau_qrt,
    "AMT_REQ_CREDIT_BUREAU_YEAR": request.amt_req_credit_bureau_year,
    }



    df_values=pd.DataFrame(dict_values,index=[0])
    logging.debug("Taille du df"+df_values.shape[0])
    logging.debug("Nombre de variables du df"+df_values.shape[1])

    prediction_result=model_logreg.predict_proba(df_values)
    logging.debug(prediction_result)

    prediction_result=prediction_result[0][1]
    logging.debug("Votre message de débogage ici")

    prediction_result=str(prediction_result.round(2))



    return PredictionResponse(prediction=prediction_result)


if __name__ == '__main__':


    uvicorn.run(app, host='127.0.0.3', port=8000)