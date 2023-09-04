import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_data():
    """

    :return:
    """
    mimic_data = pd.read_csv("./data/mimic-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv")
    mimic_data = mimic_data.drop(mimic_data.filter(regex='Unnamed').columns, axis=1)
    # print("mimic_data.head() = ", mimic_data.columns, len(mimic_data), mimic_data.dtypes)

    # assign race groups
    mimic_data["race"] = mimic_data["ethnicity"].str.lower().apply(lambda x: group_assign_race(x))

    # create a dead? column
    mimic_data["died-in-hosp?"] = np.where(mimic_data["discharge_location"] == "dead/expired", 1, 0)
    # print('mimic_data["died-in-hosp?"] values = ', sum(mimic_data["died-in-hosp?"].values.tolist()))

    # create a favorable-dl? column
    mimic_data["fav-disch-loc?"] = np.where(mimic_data["discharge_location"].isin(
        ["home", "home health care", "home with home iv providr"]), 1, 0)
    # print('mimic_data["fav-disch-loc?"] values = ', sum(mimic_data["fav-disch-loc?"].values.tolist()))

    # preprocess data before splitting it
    categorical_columns = ["agegroup", "gender", "insurance", "race"]
    model_non_categorical_features = ["age", "los-h(days)", "n-stemi?", "shock?", "c-shock?", "received-analgesic?",
                                      "received-opioid?",
                                      "received-non-opioid?", "received-opioids-only?", "received-non-opioids-only?",
                                      "received-combined-therapy?", "received-ace-inhibitor?",
                                      "received-aspirin?", "received-beta-blocker?",
                                      "received-anti-platelet?", "received-statin?"]
    outcome_features = ["died-in-hosp?", "fav-disch-loc?"]

    # convert categorical features to ohe
    categorical_values_to_drop = ["18-39", "m", "self pay", "unknown/unspecified", "unknown (default)"]
    ohe_categorical_data_df = convert_categorical_to_numerical(
        df=mimic_data[categorical_columns], categorical_variables=categorical_columns,
        to_drop=categorical_values_to_drop)
    # print("ohe_categorical_data_df = ", ohe_categorical_data_df.head(), len(ohe_categorical_data_df),
    #       ohe_categorical_data_df.columns.tolist())

    # create a df of relevant model features
    df = mimic_data[model_non_categorical_features]
    df = pd.concat([df, ohe_categorical_data_df, mimic_data[outcome_features]], axis=1)
    # print("df.head() = ", df.head(), "df cols = ", df.columns.tolist(), "no of cols =", len(df.columns.tolist()),
    #       "len(df)=", len(df))

    # split data into train test
    train, test = train_test_split(df, test_size=0.33, random_state=13)
    # print("train.head = ", train.head(), len(train), sum(train["died-in-hosp?"]), sum(train["fav-disch-loc?"]))
    # print("test.head = ", test.head(), len(test), sum(test["died-in-hosp?"]), sum(test["fav-disch-loc?"]))
    train.to_csv("./data/mimic-train.csv")
    test.to_csv("./data/mimic-test.csv")


def convert_categorical_to_numerical(df, categorical_variables, to_drop):
    """
    This function one-hot encodes categorical variables in a dataframe
    :param df: dataframe
    :param categorical_variables: a list of column names of the categorical data that need to be coded
           e.g., ["gender", "hobby"]
    :param to_drop: a list of the values of the categorical columns to be dropped. e.g. ["female", "skiing"].
           These are assumed to be the "default" values. Dropping values reduces multi-linearity,
           so by default this function assumes that some values will be dropped
    :return: a dataframe of the original categorical columns expanded using one-hot encoding
    """
    one_hot_encoded_list = []
    for i in range(len(categorical_variables)):
        one_hot_encoded_list.append(df[categorical_variables[i]].str.get_dummies().add_prefix(
            categorical_variables[i] + "-").drop(categorical_variables[i] + "-" + to_drop[i], axis=1))
    one_hot_encoded_df = pd.concat(one_hot_encoded_list, axis=1)
    return one_hot_encoded_df


def group_assign_race(x):
    """
    Helper function to assign a patient into a racial group.
    :param x: recorded ethnicities
    :return: patient race group
    """
    if ("white" in x) | ("portuguese" in x):
        return "white/caucasian-american"
    elif ("black" in x) | ("african" in x):
        return "black/african-american"
    elif "asian" in x:
        return "asian/asian-american"
    elif ("hispanic" in x) | ("latino" in x):
        return "latinx/hispanic-american"
    elif ("american indian" in x) | ("alaska native" in x) | ("alaska" in x):
        return "alaska-native/american-indian"
    elif ("hawaiian" in x) | ("pacific islander" in x):
        return "polynesian/pacific-islander"
    else:
        return "unknown/unspecified"



