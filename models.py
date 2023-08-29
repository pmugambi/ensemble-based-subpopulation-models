from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


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
    categorical_columns = ["agegroup", "gender", "insurance",
                           "race"]  # todo: think about whether including "icd9_code" is valuable
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


def flatten(l):
    """

    :param l:
    :return:
    """
    return [item for sublist in l for item in sublist]


def pre_modeling_pipeline(train_X, test_X=None):
    """

    :param train_X:
    :param test_X:
    :return:
    """
    # impute missing values
    imputer = SimpleImputer(strategy="median")
    # # imputer = SimpleImputer(strategy="mean")
    train_X = imputer.fit_transform(train_X)
    # normalize? check whether it's valuable to normalize/scale data
    # i'm doing this to buy myself the option of only working with train data until unless testing
    if test_X is not None:
        test_X = imputer.transform(test_X)
        return train_X, test_X
    else:
        return train_X


def classifiers_to_test():
    """

    :return:
    """
    dt_clf = DecisionTreeClassifier(random_state=0)
    rf_clf = RandomForestClassifier(random_state=0)
    lr_clf = LogisticRegression(random_state=0, max_iter=1500)
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=0, max_depth=1)
    mlp_clf = MLPClassifier(random_state=1, max_iter=1800)
    svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=0)
    classifiers = [dt_clf, rf_clf, lr_clf, gb_clf, mlp_clf, svm_clf, xgb_clf]
    classifier_names = ["dt", "rf", "log-reg", "gb", "mlp", "svm", "xgb"]
    return classifiers, classifier_names


def extract_cv_test_scores(cv_scores):
    """

    :param cv_scores:
    :return:
    """
    accuracy_scores = cv_scores["test_accuracy"]
    balanced_accuracy_scores = cv_scores["test_balanced_accuracy"]
    f1_scores = cv_scores["test_f1"]
    precision_scores = cv_scores["test_precision_macro"]
    recall_scores = cv_scores["test_recall_macro"]
    auc_scores = cv_scores["test_roc_auc"]
    return [accuracy_scores, accuracy_scores.mean(), balanced_accuracy_scores,
            balanced_accuracy_scores.mean(), f1_scores,
            f1_scores.mean(), precision_scores, precision_scores.mean(),
            recall_scores, recall_scores.mean(), auc_scores, auc_scores.mean()]


def predict_discharge(X_train, train_ys, X_test, test_ys, X_col_names, y_names=None, save_path_add_ons=None,
                      write_add_ons=None, plot_calibration=False, cv=2):
    """

    :param X_train:
    :param train_ys:
    :param X_test:
    :param test_ys:
    :param X_col_names:
    :param y_names:
    :param save_path_add_ons:
    :param write_add_ons:
    :param plot_calibration:
    :param cv:
    :return:
    """
    # define classifiers
    if y_names is None:
        y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]

    classifiers, classifier_names = classifiers_to_test()

    # create modeling pipeline
    scoring = ["precision_macro", "recall_macro", "accuracy", "balanced_accuracy", "f1", "roc_auc"]
    for k in range(len(train_ys)):
        print(" working on outcome: -- ", y_names[k])
        outcome = y_names[k]
        uncalibrated_scores_data = []
        sigmoid_calibration_scores_data = []
        isotonic_calibration_scores_data = []
        predictions = []
        for i in range(len(classifiers)):
            model = classifiers[i]
            classifier_name = classifier_names[i]
            print("--------*testing*-------- " + classifier_names[i])

            # fit with CV and get scores
            cv_scores = cross_validate(model, cv=cv, X=X_train, y=train_ys[k], scoring=scoring)
            uncalibrated_scores_data.append([classifier_name] + extract_cv_test_scores(cv_scores))

            # fit without CV
            uncalibrated_model = model.fit(X_train, y=train_ys[k])

            # calibrate models
            # fit with CV and get scores
            sigmoid_calibrated_cv_scores = cross_validate(CalibratedClassifierCV(model, cv=2, method='sigmoid'),
                                                          X_train, train_ys[k], scoring=scoring)
            sigmoid_calibration_scores_data.append([classifier_name] + extract_cv_test_scores(
                sigmoid_calibrated_cv_scores))
            isotonic_calibrated_cv_scores = cross_validate(CalibratedClassifierCV(model, cv=2, method='isotonic'),
                                                           X_train, train_ys[k], scoring=scoring)
            isotonic_calibration_scores_data.append([classifier_name] + extract_cv_test_scores(
                isotonic_calibrated_cv_scores))

            # fit without CV
            sigmoid_calibrated_model = CalibratedClassifierCV(model, cv=2, method='sigmoid').fit(X_train, train_ys[k])
            isotonic_calibrated_model = CalibratedClassifierCV(model, cv=2, method='isotonic').fit(X_train, train_ys[k])

            calibration_dir = "./data/analysis/models/single-model/training/reliability-plots/"

            # plot and save calibration graphs for training set
            if plot_calibration:
                plot_calibration_curve(model=uncalibrated_model, X=X_train, y=train_ys[k],
                                       save_path=calibration_dir + outcome + "/",
                                       save_name=classifier_name + "-uncalibrated-reliability-plot.png")
                plot_calibration_curve(model=sigmoid_calibrated_model, X=X_train, y=train_ys[k],
                                       save_path=calibration_dir + outcome + "/",
                                       save_name=classifier_name + "-sigmoid-calibrated-reliability-plot.png")
                plot_calibration_curve(model=isotonic_calibrated_model, X=X_train, y=train_ys[k],
                                       save_path=calibration_dir + outcome + "/",
                                       save_name=classifier_name + "-isotonic-calibrated-reliability-plot.png")
            # predict todo: predict on X-test
            uncalibrated_y_pred = uncalibrated_model.predict(X_test)
            sigmoid_calibrated_y_pred = sigmoid_calibrated_model.predict(X_test)
            isotonic_calibrated_y_pred = isotonic_calibrated_model.predict(X_test)
            predictions.append([uncalibrated_y_pred, sigmoid_calibrated_y_pred,
                                isotonic_calibrated_y_pred])
        uncalibrated_results_df = pd.DataFrame(data=uncalibrated_scores_data,
                                               columns=["model-name", "cv-accuracy-raw", "cv-accuracy-mean",
                                                        "cv-balanced-accuracy-raw", "cv-balanced-accuracy-mean",
                                                        "cv-f1-raw",
                                                        "cv-f1-mean", "cv-precision-raw", "cv-precision-mean",
                                                        "cv-recall-raw",
                                                        "cv-recall-mean", "cv-auc-raw", "cv-auc-mean"])
        sigmoid_calibrated_results_df = pd.DataFrame(data=sigmoid_calibration_scores_data,
                                                     columns=["model-name", "cv-accuracy-raw", "cv-accuracy-mean",
                                                              "cv-balanced-accuracy-raw", "cv-balanced-accuracy-mean",
                                                              "cv-f1-raw",
                                                              "cv-f1-mean", "cv-precision-raw", "cv-precision-mean",
                                                              "cv-recall-raw",
                                                              "cv-recall-mean", "cv-auc-raw", "cv-auc-mean"])
        isotonic_calibrated_results_df = pd.DataFrame(data=isotonic_calibration_scores_data,
                                                      columns=["model-name", "cv-accuracy-raw", "cv-accuracy-mean",
                                                               "cv-balanced-accuracy-raw", "cv-balanced-accuracy-mean",
                                                               "cv-f1-raw",
                                                               "cv-f1-mean", "cv-precision-raw", "cv-precision-mean",
                                                               "cv-recall-raw",
                                                               "cv-recall-mean", "cv-auc-raw", "cv-auc-mean"])

        names = ["dt-preds", "dt-sig-c-preds", "dt-iso-c-preds",
                 "rf-preds", "rf-sig-c-preds", "rf-iso-c-preds",
                 "log-reg-preds", "log-reg-sig-c-preds", "log-reg-iso-c-preds",
                 "gb-preds", "gb-sig-c-preds", "gb-iso-c-preds",
                 "mlp-preds", "mlp-sig-c-preds", "mlp-iso-c-preds",
                 "svm-preds", "svm-sig-c-preds", "svm-iso-c-preds",
                 "xgb-preds", "xgb-sig-c-preds", "xgb-iso-c-preds"]
        predictions_df = pd.DataFrame(data=flatten(predictions))
        # print("predictions_df.head 1 = ", predictions_df.head())
        predictions_df = predictions_df.T
        predictions_df.columns = names
        predictions_df["true-y"] = test_ys[k]
        test_predictions_df = pd.concat([pd.DataFrame(X_test), predictions_df], axis=1)
        test_predictions_df.columns = X_col_names + predictions_df.columns.tolist()

        save_dir = "./data/analysis/models/single-model/training/" + outcome + "/"
        if save_path_add_ons is not None:
            save_dir += save_path_add_ons
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if write_add_ons is None:
            uncalibrated_results_df.to_csv(save_dir + "cv-training-results-for-" + outcome + ".csv")
            sigmoid_calibrated_results_df.to_csv(save_dir + "sigmoid-calibrated-cv-training-results-for-"
                                                 + outcome + ".csv")
            isotonic_calibrated_results_df.to_csv(save_dir + "isotonic-calibrated-cv-training-results-for-"
                                                  + outcome + ".csv")
        else:
            uncalibrated_results_df.to_csv(save_dir + "cv-training-results-for-" + outcome + write_add_ons + ".csv")
            sigmoid_calibrated_results_df.to_csv(save_dir + "sigmoid-calibrated-cv-training-results-for-" + outcome
                                                 + write_add_ons + ".csv")
            isotonic_calibrated_results_df.to_csv(save_dir + "isotonic-calibrated-cv-training-results-for-" + outcome
                                                  + write_add_ons + ".csv")

        preds_save_dir = "./data/analysis/models/single-model/prediction/" + outcome + "/"
        if save_path_add_ons is not None:
            preds_save_dir += save_path_add_ons
        Path(preds_save_dir).mkdir(parents=True, exist_ok=True)
        if write_add_ons is None:
            test_predictions_df.to_csv(preds_save_dir + "features-and-prediction-results-for-" + outcome + ".csv")
        else:
            test_predictions_df.to_csv(preds_save_dir + "features-and-prediction-results-for-" + outcome
                                       + write_add_ons + ".csv")


def predict_discharge_on_entire_population():
    """

    :return:
    """
    train_df = pd.read_csv("./data/mimic-train.csv")
    test_df = pd.read_csv("./data/mimic-test.csv")
    cols, train_X, test_X, train_ys, test_ys = process_training_data(train_df, test_df)
    print("died-in-hosp% = ", np.round((train_ys[0].sum() / len(train_df) * 100)), train_ys[0].sum())
    print("fav-disch-loc% = ", np.round((train_ys[1].sum() / len(train_df) * 100)), train_ys[1].sum())
    predict_discharge(X_train=train_X, X_test=test_X,
                      train_ys=train_ys, test_ys=test_ys,
                      y_names=["y1-in-hosp-mortality", "y2-favorable-discharge-loc"],
                      X_col_names=cols,
                      save_path_add_ons="entire-population/", cv=3,
                      write_add_ons="-entire-population", plot_calibration=True)


def predict_discharge_for_sub_populations():
    """

    :return:
    """
    # by sex
    train_df = pd.read_csv("./data/mimic-train.csv")
    test_df = pd.read_csv("./data/mimic-test.csv")
    piis = ["sex", "race", "insurance", "age-group"]
    for pii in piis:
        print("****** fitting a single model for each subgroup. now fitting models for: " + pii + " ****")
        if pii == "sex":
            male_train_X, female_train_X, male_train_ys, female_train_ys, male_X_cols, female_X_cols = obtain_subgroups(
                task="train", pii=pii, df=train_df)
            male_test_X, female_test_X, male_test_ys, female_test_ys, _, _ = obtain_subgroups(
                task="train", pii=pii, df=test_df, preprocess=False)
            # build models
            predict_discharge(X_train=male_train_X, train_ys=male_train_ys, X_test=male_test_X, test_ys=male_test_ys,
                              X_col_names=male_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-male-patients")
            predict_discharge(X_train=female_train_X, train_ys=female_train_ys, X_test=female_test_X,
                              test_ys=female_test_ys, X_col_names=female_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-female-patients")
        elif pii == "race":
            white_train_X, non_white_train_X, white_train_ys, non_white_train_ys, white_X_cols, \
            non_white_X_cols = obtain_subgroups(task="train", pii=pii, df=train_df)
            white_test_X, non_white_test_X, white_test_ys, non_white_test_ys, _, _ = \
                obtain_subgroups(task="train", pii=pii, df=test_df, preprocess=False)
            # build models
            predict_discharge(X_train=white_train_X, train_ys=white_train_ys, X_test=white_test_X,
                              test_ys=white_test_ys, X_col_names=white_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-white-patients")
            predict_discharge(X_train=non_white_train_X, train_ys=non_white_train_ys, X_test=non_white_test_X,
                              test_ys=non_white_test_ys, X_col_names=non_white_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-non-white-patients")
        elif pii == "insurance":
            private_train_X, government_train_X, private_train_ys, government_train_ys, private_X_cols, \
            government_X_cols = obtain_subgroups(
                task="train", pii=pii, df=train_df)
            private_test_X, government_test_X, private_test_ys, government_test_ys, _, _ = obtain_subgroups(
                task="train", pii=pii, df=test_df, preprocess=False)
            # build models
            predict_discharge(X_train=private_train_X, train_ys=private_train_ys, X_test=private_test_X,
                              test_ys=private_test_ys, X_col_names=private_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-private-patients")
            predict_discharge(X_train=government_train_X, train_ys=government_train_ys, X_test=government_test_X,
                              test_ys=government_test_ys, X_col_names=government_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-government-patients")
        elif pii == "age-group":
            forties_train_X, fifties_train_X, sixties_train_X, seventies_train_X, eighty_and_over_train_X, \
            forties_train_ys, fifties_train_ys, sixties_train_ys, seventies_train_ys, eighty_and_over_train_ys, \
            forties_X_cols, fifties_X_cols, sixties_X_cols, seventies_X_cols, \
            eighty_and_over_X_cols = obtain_subgroups(task="train", pii=pii, df=train_df)
            forties_test_X, fifties_test_X, sixties_test_X, seventies_test_X, eighty_and_over_test_X, \
            forties_test_ys, fifties_test_ys, sixties_test_ys, seventies_test_ys, eighty_and_over_test_ys, \
            _, _, _, _, _ = obtain_subgroups(task="train", pii=pii, df=test_df, preprocess=False)
            # build models
            predict_discharge(X_train=forties_train_X, train_ys=forties_train_ys, X_test=forties_test_X,
                              test_ys=forties_test_ys, X_col_names=forties_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-forties-patients")
            predict_discharge(X_train=fifties_train_X, train_ys=fifties_train_ys, X_test=fifties_test_X,
                              test_ys=fifties_test_ys, X_col_names=fifties_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-fifties-patients")
            predict_discharge(X_train=sixties_train_X, train_ys=sixties_train_ys, X_test=sixties_test_X,
                              test_ys=sixties_test_ys, X_col_names=sixties_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-sixties-patients")
            predict_discharge(X_train=seventies_train_X, train_ys=seventies_train_ys, X_test=seventies_test_X,
                              test_ys=seventies_test_ys, X_col_names=seventies_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-seventies-patients")
            predict_discharge(X_train=eighty_and_over_train_X, train_ys=eighty_and_over_train_ys,
                              X_test=eighty_and_over_test_X, test_ys=eighty_and_over_test_ys,
                              X_col_names=eighty_and_over_X_cols,
                              save_path_add_ons="subgroup/" + pii + "/", write_add_ons="-eighty-and-over-patients")


def obtain_best_model(level, y_name, pii=None):
    base_dir = "./data/analysis/models/single-model/training/" + y_name + "/" + level + "/"
    if level == "entire-population":
        cv_training_results_df = pd.read_csv(base_dir + "cv-training-results-for-" + y_name + "-" + level + ".csv")
        # print("cv_training_results_df = ", cv_training_results_df.head())
        best_clf = top_model_by_metric(df=cv_training_results_df)
        return best_clf
    elif level == "subgroup":  # todo: this code could be optimized to avoid repetition
        if pii == "sex":
            male_results_df = pd.read_csv(base_dir + "sex/cv-training-results-for-" + y_name + "-male-patients.csv")
            female_results_df = pd.read_csv(base_dir + "sex/cv-training-results-for-" + y_name + "-female-patients.csv")
            best_clf_male = top_model_by_metric(df=male_results_df)
            best_clf_female = top_model_by_metric(df=female_results_df)
            print("best male clf = ", best_clf_male)
            return best_clf_male, best_clf_female
        elif pii == "race":
            white_results_df = pd.read_csv(base_dir + "race/cv-training-results-for-" + y_name + "-white-patients.csv")
            non_white_results_df = pd.read_csv(base_dir + "race/cv-training-results-for-" + y_name +
                                               "-non-white-patients.csv")
            best_clf_white = top_model_by_metric(df=white_results_df)
            best_clf_non_white = top_model_by_metric(df=non_white_results_df)
            print("best white clf = ", best_clf_white)
            return best_clf_white, best_clf_non_white
        elif pii == "insurance":
            private_results_df = pd.read_csv(base_dir + "insurance/cv-training-results-for-" + y_name +
                                             "-private-patients.csv")
            government_results_df = pd.read_csv(base_dir + "insurance/cv-training-results-for-" + y_name +
                                                "-government-patients.csv")
            best_clf_private = top_model_by_metric(df=private_results_df)
            best_clf_government = top_model_by_metric(df=government_results_df)
            print("best private clf = ", best_clf_private)
            return best_clf_private, best_clf_government
        elif pii == "age-group":
            forties_results_df = pd.read_csv(base_dir + "age-group/cv-training-results-for-" + y_name +
                                             "-forties-patients.csv")
            fifties_results_df = pd.read_csv(base_dir + "age-group/cv-training-results-for-" + y_name +
                                             "-fifties-patients.csv")
            sixties_results_df = pd.read_csv(base_dir + "age-group/cv-training-results-for-" + y_name +
                                             "-sixties-patients.csv")
            seventies_results_df = pd.read_csv(base_dir + "age-group/cv-training-results-for-" + y_name +
                                               "-seventies-patients.csv")
            eighty_and_over_results_df = pd.read_csv(base_dir + "age-group/cv-training-results-for-" + y_name +
                                                     "-eighty-and-over-patients.csv")
            best_clf_forties = top_model_by_metric(df=forties_results_df)
            best_clf_fifties = top_model_by_metric(df=fifties_results_df)
            best_clf_sixties = top_model_by_metric(df=sixties_results_df)
            best_clf_seventies = top_model_by_metric(df=seventies_results_df)
            best_clf_eighty_and_over = top_model_by_metric(df=eighty_and_over_results_df)
            print("best forties clf = ", best_clf_forties)
            return best_clf_forties, best_clf_fifties, best_clf_sixties, best_clf_seventies, best_clf_eighty_and_over
    else:
        return ValueError("The accepted values for pii are: sex, race, insurance, and age-group")


def top_model_by_metric(df, metric="f1"):
    # best model by f1 score
    if metric == "f1":
        best = df[df["cv-f1-mean"] == df["cv-f1-mean"].max()]["model-name"].values.tolist()[0]
    elif metric == "balanced-accuracy":
        best = \
            df[df["cv-balanced-accuracy-mean"] == df["cv-balanced-accuracy-mean"].max()]["model-name"].values.tolist()[
                0]  # taking only the best 1
    elif metric == "accuracy":
        best = df[df["cv-accuracy-mean"] == df["cv-accuracy-mean"].max()]["model-name"].values.tolist()[0]
    else:
        return ValueError("this metric wasn't computed during training")
    return best.strip()


def bottom_up_ensemble(estimators, train_X, train_y, test_X, test_y, X_columns, ensemble_cv_save_name,
                       ensemble_predictions_save_name):
    # create our voting classifier, inputting our models
    ensemble_hard = VotingClassifier(estimators, voting="hard")
    ensemble_soft = VotingClassifier(estimators, voting="soft")

    ensemble_hard.fit(train_X, train_y)
    ensemble_soft.fit(train_X, train_y)
    # print("success! getting here")
    ensemble_soft_cv_scores = cross_validate(ensemble_soft, X=train_X, y=train_y,
                                             scoring=["precision_macro", "recall_macro", "accuracy",
                                                      "balanced_accuracy", "f1", "roc_auc"])
    # get ensemble_soft scores
    ensemble_soft_accuracy_scores = ensemble_soft_cv_scores["test_accuracy"]
    ensemble_soft_balanced_accuracy_scores = ensemble_soft_cv_scores["test_balanced_accuracy"]
    ensemble_soft_f1_scores = ensemble_soft_cv_scores["test_f1"]
    ensemble_soft_precision_scores = ensemble_soft_cv_scores["test_precision_macro"]
    ensemble_soft_recall_scores = ensemble_soft_cv_scores["test_recall_macro"]
    ensemble_soft_auc_scores = ensemble_soft_cv_scores["test_roc_auc"]
    # print("average acc on e_y1_train = ", ensemble_soft_accuracy_scores.mean())
    # print("average f1 on e_y1_train = ", ensemble_soft_f1_scores.mean())
    # print("average precision on e_y1_train = ", ensemble_soft_precision_scores.mean())
    # print("average recall on e_y1_train = ", ensemble_soft_recall_scores.mean())
    # print("average auc on e_y1_train = ", ensemble_soft_auc_scores.mean())

    ensemble_soft_results_df = pd.DataFrame(data=[[ensemble_soft_accuracy_scores,
                                                   ensemble_soft_accuracy_scores.mean(),
                                                   ensemble_soft_balanced_accuracy_scores,
                                                   ensemble_soft_balanced_accuracy_scores.mean(),
                                                   ensemble_soft_f1_scores, ensemble_soft_f1_scores.mean(),
                                                   ensemble_soft_precision_scores,
                                                   ensemble_soft_precision_scores.mean(),
                                                   ensemble_soft_recall_scores,
                                                   ensemble_soft_recall_scores.mean(), ensemble_soft_auc_scores,
                                                   ensemble_soft_auc_scores.mean()]],
                                            columns=["cv-accuracy-raw", "cv-accuracy-mean",
                                                     "cv-balanced-accuracy-raw",
                                                     "cv-balanced-accuracy-mean", "cv-f1-raw", "cv-f1-mean",
                                                     "cv-precision-raw", "cv-precision-mean", "cv-recall-raw",
                                                     "cv-recall-mean", "cv-auc-raw", "cv-auc-mean"])

    ensemble_soft_results_df.to_csv(ensemble_cv_save_name)

    # predict
    ensemble_hard_train_preds = ensemble_hard.predict(test_X)
    ensemble_soft_train_preds = ensemble_soft.predict(test_X)

    # print("true y == ys[i] ", test_y)
    # print("ensemble-hard preds = ", ensemble_hard_train_preds)
    # print("ensemble-soft preds = ", ensemble_soft_train_preds)

    predictions_df = pd.DataFrame(data=[test_y.tolist(), ensemble_hard_train_preds.tolist(),
                                        ensemble_soft_train_preds.tolist()])
    predictions_df = predictions_df.T
    predictions_df.columns = ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
    test_predictions_df = pd.concat([pd.DataFrame(test_X), predictions_df], axis=1)
    test_predictions_df.columns = X_columns + ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
    # print("test predictions df.head = ", test_predictions_df.head())
    test_predictions_df.to_csv(ensemble_predictions_save_name)


def ensemble_on_entire_population():
    """

    :return:
    """
    # take best model for each group, then add them into a voting classifier,
    # then fit the classifier on entire train data
    outcomes = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    classifiers, classifiers_names = classifiers_to_test()

    estimators = {
        "y1-in-hosp-mortality": [("rf", classifiers[classifiers_names.index("rf")]),
                                 ("xgb", classifiers[classifiers_names.index("xgb")]),
                                 ("mlp", classifiers[classifiers_names.index("mlp")])],
        # todo: play with this and see effect of more models. for now, just adding the top 3 models for each
        "y2-favorable-discharge-loc": [("gb", classifiers[classifiers_names.index("gb")]),
                                       ("log-reg", classifiers[classifiers_names.index("log-reg")]),
                                       ("mlp", classifiers[classifiers_names.index("mlp")])]
    }

    train_df = pd.read_csv("./data/mimic-train.csv")
    train_df = train_df.drop(train_df.filter(regex='Unnamed').columns, axis=1)
    test_df = pd.read_csv("./data/mimic-test.csv")
    test_df = test_df.drop(test_df.filter(regex='Unnamed').columns, axis=1)
    cols, train_X, test_X, train_ys, test_ys = process_training_data(train_df=train_df, test_df=test_df)

    for i in range(len(outcomes)):  # todo: this portion of the code can be optimized to avoid repeat of similar logic.
        y_estimators = estimators[outcomes[i]]
        # by sex
        male_X, female_X, male_ys, female_ys, _, _ = obtain_subgroups(
            task="train", pii="sex", df=train_df)
        male_best_clf, female_best_clf = obtain_best_model(level="subgroup", y_name=outcomes[0], pii="sex")
        m_clf = classifiers[classifiers_names.index(male_best_clf)]
        f_clf = classifiers[classifiers_names.index(female_best_clf)]

        if True in ["female-best" in x for x in y_estimators]:
            y_estimators.remove(estimators[["female-best" in x for x in estimators].index(True)])
            y_estimators.remove(estimators[["male-best" in x for x in estimators].index(True)])
            y_estimators.append(('male-best', m_clf.fit(male_X, male_ys[i])))
            y_estimators.append(("female-best", f_clf.fit(female_X, female_ys[i])))
        else:
            y_estimators.append(('male-best', m_clf.fit(male_X, male_ys[i])))
            y_estimators.append(("female-best", f_clf.fit(female_X, female_ys[i])))

        sex_train_save_dir = "./data/analysis/models/ensemble/training/" + outcomes[i] + \
                             "/ensemble-on-entire-population/sex/"
        sex_pred_save_dir = "./data/analysis/models/ensemble/prediction/" + outcomes[i] + \
                            "/ensemble-on-entire-population/sex/"
        Path(sex_train_save_dir).mkdir(parents=True, exist_ok=True)
        Path(sex_pred_save_dir).mkdir(parents=True, exist_ok=True)
        bottom_up_ensemble(train_X=train_X, train_y=train_ys[i], test_X=test_X, test_y=test_ys[i],
                           estimators=y_estimators, X_columns=cols,
                           ensemble_cv_save_name=sex_train_save_dir + "cv-results-for-ensemble-soft-by-sex.csv",
                           ensemble_predictions_save_name=sex_pred_save_dir + "features-and-predictions-of-ensemble-by"
                                                                              "-sex.csv")
        # by race
        white_X, non_white_X, white_ys, non_white_ys, _, _ = obtain_subgroups(
            task="train", pii="race", df=train_df)
        white_best_clf, non_white_best_clf = obtain_best_model(level="subgroup", y_name=outcomes[0], pii="race")
        white_clf = classifiers[classifiers_names.index(white_best_clf)]
        non_white_clf = classifiers[classifiers_names.index(non_white_best_clf)]

        if True in ["white-best" in x for x in y_estimators]:
            y_estimators.remove(estimators[["white-best" in x for x in estimators].index(True)])
            y_estimators.remove(estimators[["non-white-best" in x for x in estimators].index(True)])
            y_estimators.append(('white-best', white_clf.fit(white_X, white_ys[i])))
            y_estimators.append(("non-white-best", non_white_clf.fit(non_white_X, non_white_ys[i])))
        else:
            y_estimators.append(('white-best', white_clf.fit(white_X, white_ys[i])))
            y_estimators.append(("non-white-best", non_white_clf.fit(non_white_X, non_white_ys[i])))

        race_train_save_dir = "./data/analysis/models/ensemble/training/" + outcomes[i] + \
                              "/ensemble-on-entire-population/race/"
        race_pred_save_dir = "./data/analysis/models/ensemble/prediction/" + outcomes[i] + \
                             "/ensemble-on-entire-population/race/"
        Path(race_train_save_dir).mkdir(parents=True, exist_ok=True)
        Path(race_pred_save_dir).mkdir(parents=True, exist_ok=True)
        bottom_up_ensemble(train_X=train_X, train_y=train_ys[i], test_X=test_X, test_y=test_ys[i],
                           estimators=y_estimators, X_columns=cols,
                           ensemble_cv_save_name=race_train_save_dir + "cv-results-for-ensemble-soft-by-race.csv",
                           ensemble_predictions_save_name=race_pred_save_dir + "features-and-predictions-of-ensemble-by"
                                                                               "-race.csv")

        # by insurance
        private_X, government_X, private_ys, government_ys, _, _ = obtain_subgroups(
            task="train", pii="insurance", df=train_df)
        private_best_clf, government_best_clf = obtain_best_model(level="subgroup", y_name=outcomes[0], pii="insurance")
        private_clf = classifiers[classifiers_names.index(private_best_clf)]
        government_clf = classifiers[classifiers_names.index(government_best_clf)]

        if True in ["private-best" in x for x in y_estimators]:
            y_estimators.remove(estimators[["private-best" in x for x in estimators].index(True)])
            y_estimators.remove(estimators[["government-best" in x for x in estimators].index(True)])
            y_estimators.append(('private-best', private_clf.fit(private_X, private_ys[i])))
            y_estimators.append(("government-best", government_clf.fit(government_X, government_ys[i])))
        else:
            y_estimators.append(('private-best', private_clf.fit(private_X, private_ys[i])))
            y_estimators.append(("government-best", government_clf.fit(government_X, government_ys[i])))

        insurance_train_save_dir = "./data/analysis/models/ensemble/training/" + outcomes[i] + \
                                   "/ensemble-on-entire-population/insurance/"
        insurance_pred_save_dir = "./data/analysis/models/ensemble/prediction/" + outcomes[i] + \
                                  "/ensemble-on-entire-population/insurance/"
        Path(insurance_train_save_dir).mkdir(parents=True, exist_ok=True)
        Path(insurance_pred_save_dir).mkdir(parents=True, exist_ok=True)
        bottom_up_ensemble(train_X=train_X, train_y=train_ys[i], test_X=test_X, test_y=test_ys[i],
                           estimators=y_estimators, X_columns=cols,
                           ensemble_cv_save_name=insurance_train_save_dir + "cv-results-for-ensemble-soft-by"
                                                                            "-insurance.csv",
                           ensemble_predictions_save_name=insurance_pred_save_dir + "features-and-predictions-of"
                                                                                    "-ensemble-by "
                                                                                    "-insurance.csv")

        # by age-group
        forties_X, fifties_X, sixties_X, seventies_X, eight_and_over_X, forties_ys, fifties_ys, sixties_ys, \
        seventies_ys, eight_and_over_ys, _, _, _, _, _ = obtain_subgroups(task="train", pii="age-group",
                                                                          df=train_df)
        forties_best_clf, fifties_best_clf, sixties_best_clf, seventies_best_clf, eight_and_over_best_clf = \
            obtain_best_model(level="subgroup", y_name=outcomes[0], pii="age-group")
        forties_clf = classifiers[classifiers_names.index(forties_best_clf)]
        fifties_clf = classifiers[classifiers_names.index(fifties_best_clf)]
        sixties_clf = classifiers[classifiers_names.index(sixties_best_clf)]
        seventies_clf = classifiers[classifiers_names.index(seventies_best_clf)]
        eight_and_over_clf = classifiers[classifiers_names.index(eight_and_over_best_clf)]

        if True in ["forties-best" in x for x in y_estimators]:
            y_estimators.remove(estimators[["forties-best" in x for x in estimators].index(True)])
            y_estimators.remove(estimators[["fifties-best" in x for x in estimators].index(True)])
            y_estimators.remove(estimators[["sixties-best" in x for x in estimators].index(True)])
            y_estimators.remove(estimators[["seventies-best" in x for x in estimators].index(True)])
            y_estimators.remove(estimators[["eighty-and-over-best" in x for x in estimators].index(True)])

            y_estimators.append(('forties-best', forties_clf.fit(forties_X, forties_ys[i])))
            y_estimators.append(('fifties-best', fifties_clf.fit(fifties_X, fifties_ys[i])))
            y_estimators.append(('sixties-best', sixties_clf.fit(sixties_X, sixties_ys[i])))
            y_estimators.append(('seventies-best', seventies_clf.fit(seventies_X, seventies_ys[i])))
            y_estimators.append(('eight-and-over-best', eight_and_over_clf.fit(eight_and_over_X, eight_and_over_ys[i])))

        else:
            y_estimators.append(('forties-best', forties_clf.fit(forties_X, forties_ys[i])))
            y_estimators.append(('fifties-best', fifties_clf.fit(fifties_X, fifties_ys[i])))
            y_estimators.append(('sixties-best', sixties_clf.fit(sixties_X, sixties_ys[i])))
            y_estimators.append(('seventies-best', seventies_clf.fit(seventies_X, seventies_ys[i])))
            y_estimators.append(('eight-and-over-best', eight_and_over_clf.fit(eight_and_over_X, eight_and_over_ys[i])))

        insurance_train_save_dir = "./data/analysis/models/ensemble/training/" + outcomes[i] + \
                                   "/ensemble-on-entire-population/age-group/"
        insurance_pred_save_dir = "./data/analysis/models/ensemble/prediction/" + outcomes[i] + \
                                  "/ensemble-on-entire-population/age-group/"
        Path(insurance_train_save_dir).mkdir(parents=True, exist_ok=True)
        Path(insurance_pred_save_dir).mkdir(parents=True, exist_ok=True)
        bottom_up_ensemble(train_X=train_X, train_y=train_ys[i], test_X=test_X, test_y=test_ys[i],
                           estimators=y_estimators, X_columns=cols,
                           ensemble_cv_save_name=insurance_train_save_dir + "cv-results-for-ensemble-soft-by"
                                                                            "-age-group.csv",
                           ensemble_predictions_save_name=insurance_pred_save_dir + "features-and-predictions-of"
                                                                                    "-ensemble-by "
                                                                                    "-age-group.csv")


def ensemble_top_down(estimators, train_Xs, test_Xs, train_group_ys, test_group_ys, group_names, i, cols,
                      train_save_dir, pred_save_dir):
    """

    :param estimators:
    :param Xs:
    :param group_ys:
    :param group_names:
    :param i:
    :param cols:
    :param train_save_dir:
    :param pred_save_dir:
    :return:
    """
    for j in range(len(train_Xs)):
        ensemble_hard = VotingClassifier(estimators, voting="hard")
        ensemble_soft = VotingClassifier(estimators, voting="soft")

        ensemble_hard.fit(train_Xs[j], train_group_ys[j][i])
        # print("group = ", group_names[j], " ensemble hard fit = ", ensemble_hard)
        ensemble_soft.fit(train_Xs[j], train_group_ys[j][i])

        ensemble_soft_cv_scores = cross_validate(ensemble_soft, X=train_Xs[j], y=train_group_ys[j][i],
                                                 scoring=["precision_macro", "recall_macro", "accuracy",
                                                          "balanced_accuracy", "f1", "roc_auc"])
        ensemble_soft_accuracy_scores = ensemble_soft_cv_scores["test_accuracy"]
        ensemble_soft_balanced_accuracy_scores = ensemble_soft_cv_scores["test_balanced_accuracy"]
        ensemble_soft_f1_scores = ensemble_soft_cv_scores["test_f1"]
        ensemble_soft_precision_scores = ensemble_soft_cv_scores["test_precision_macro"]
        ensemble_soft_recall_scores = ensemble_soft_cv_scores["test_recall_macro"]
        ensemble_soft_auc_scores = ensemble_soft_cv_scores["test_roc_auc"]
        # print("average acc on e_y1_train = ", ensemble_soft_accuracy_scores.mean())
        # print("average f1 on e_y1_train = ", ensemble_soft_f1_scores.mean())
        # print("average precision on e_y1_train = ", ensemble_soft_precision_scores.mean())
        # print("average recall on e_y1_train = ", ensemble_soft_recall_scores.mean())
        # print("average auc on e_y1_train = ", ensemble_soft_auc_scores.mean())

        ensemble_soft_results_df = pd.DataFrame(data=[[ensemble_soft_accuracy_scores,
                                                       ensemble_soft_accuracy_scores.mean(),
                                                       ensemble_soft_balanced_accuracy_scores,
                                                       ensemble_soft_balanced_accuracy_scores.mean(),
                                                       ensemble_soft_f1_scores, ensemble_soft_f1_scores.mean(),
                                                       ensemble_soft_precision_scores,
                                                       ensemble_soft_precision_scores.mean(),
                                                       ensemble_soft_recall_scores,
                                                       ensemble_soft_recall_scores.mean(),
                                                       ensemble_soft_auc_scores,
                                                       ensemble_soft_auc_scores.mean()]],
                                                columns=["cv-accuracy-raw", "cv-accuracy-mean",
                                                         "cv-balanced-accuracy-raw",
                                                         "cv-balanced-accuracy-mean", "cv-f1-raw", "cv-f1-mean",
                                                         "cv-precision-raw", "cv-precision-mean", "cv-recall-raw",
                                                         "cv-recall-mean", "cv-auc-raw", "cv-auc-mean"])

        ensemble_soft_results_df.to_csv(train_save_dir + group_names[j] + "-cv-results-for-ensemble-soft.csv")
        # # predict # todo: change this to test set
        ensemble_hard_train_preds = ensemble_hard.predict(test_Xs[j])
        ensemble_soft_train_preds = ensemble_soft.predict(test_Xs[j])

        predictions_df = pd.DataFrame(data=[test_group_ys[j][i].tolist(), ensemble_hard_train_preds.tolist(),
                                            ensemble_soft_train_preds.tolist()])
        predictions_df = predictions_df.T
        predictions_df.columns = ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
        test_predictions_df = pd.concat([pd.DataFrame(test_Xs[j]), predictions_df], axis=1)
        test_predictions_df.columns = cols + ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
        print(group_names[j] + " test predictions df.head = ", test_predictions_df.head())
        test_predictions_df.to_csv(pred_save_dir + group_names[j] + "-features-and-predictions-of-ensemble.csv")


def ensemble_on_subpopulations():
    """

    :return:
    """
    outcomes = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    classifiers, classifiers_names = classifiers_to_test()

    estimators = {
        "y1-in-hosp-mortality": [("rf", classifiers[classifiers_names.index("rf")]),
                                 ("mlp", classifiers[classifiers_names.index("mlp")]),
                                 ("gb", classifiers[classifiers_names.index("gb")])],
        "y2-favorable-discharge-loc": [("log-reg", classifiers[classifiers_names.index("log-reg")]),
                                       ("mlp", classifiers[classifiers_names.index("mlp")]),
                                       ("svm", classifiers[classifiers_names.index("svm")])]
    }  # todo: right now picking 2nd-4th best models for the entire population.
    #       Think whether it makes sense to pick best models by subpopulations

    train_df = pd.read_csv("./data/mimic-train.csv")
    train_df = train_df.drop(train_df.filter(regex='Unnamed').columns, axis=1)
    test_df = pd.read_csv("./data/mimic-test.csv")
    test_df = test_df.drop(test_df.filter(regex='Unnamed').columns, axis=1)
    cols, train_X, ys = process_training_data(train_df)

    for i in range(len(outcomes)):
        y_estimators = estimators[outcomes[i]]
        best_clf = obtain_best_model(level="entire-population", y_name=outcomes[i])
        print("best_clf = ", best_clf)

        # fit best model on entire dataset
        fitted_clf = classifiers[classifiers_names.index(best_clf)].fit(train_X, ys[i])

        # add fitted clf to estimators
        if True in ["best-population-clf" in x for x in estimators]:
            y_estimators.remove(estimators[["best-population-clf" in x for x in estimators].index(True)])
            y_estimators.append(("best-population-clf", fitted_clf))
        else:
            y_estimators.append(("best-population-clf", fitted_clf))

        # fit sub-groups
        # by sex # todo: this code can be refactored so that repetition is avoided
        male_train_X, female_train_X, male_train_ys, female_train_ys, _, _ = obtain_subgroups(task="train",
                                                                                              pii="sex", df=train_df)
        male_test_X, female_test_X, male_test_ys, female_test_ys, _, _ = obtain_subgroups(task="train",
                                                                                          pii="sex", df=test_df,
                                                                                          preprocess=False)
        sex_train_save_dir = "./data/analysis/models/ensemble/training/" + outcomes[i] + "/ensemble-on-subgroup/sex/"
        sex_pred_save_dir = "./data/analysis/models/ensemble/prediction/" + outcomes[i] + "/ensemble-on-subgroup/sex/"
        Path(sex_train_save_dir).mkdir(parents=True, exist_ok=True)
        Path(sex_pred_save_dir).mkdir(parents=True, exist_ok=True)
        ensemble_top_down(estimators=y_estimators, train_Xs=[male_train_X, female_train_X],
                          test_Xs=[male_test_X, female_test_X], train_group_ys=[male_train_ys, female_train_ys],
                          test_group_ys=[male_test_ys, female_test_ys],
                          group_names=["male", "female"], i=i, cols=cols, train_save_dir=sex_train_save_dir,
                          pred_save_dir=sex_pred_save_dir)

        # by race
        white_train_X, non_white_train_X, white_train_ys, non_white_train_ys, _, _ = obtain_subgroups(
            task="train", pii="race", df=train_df)
        white_test_X, non_white_test_X, white_test_ys, non_white_test_ys, _, _ = obtain_subgroups(
            task="train", pii="race", df=test_df, preprocess=False)
        race_train_save_dir = "./data/analysis/models/ensemble/training/" + outcomes[i] + "/ensemble-on-subgroup/race/"
        race_pred_save_dir = "./data/analysis/models/ensemble/prediction/" + outcomes[i] + "/ensemble-on-subgroup" \
                                                                                           "/race/"
        Path(race_train_save_dir).mkdir(parents=True, exist_ok=True)
        Path(race_pred_save_dir).mkdir(parents=True, exist_ok=True)
        ensemble_top_down(estimators=y_estimators, train_Xs=[white_train_X, non_white_train_X],
                          train_group_ys=[white_train_ys, non_white_train_ys], test_Xs=[white_test_X, non_white_test_X],
                          test_group_ys=[white_test_ys, non_white_test_ys],
                          group_names=["white", "non-white"], i=i, cols=cols, train_save_dir=race_train_save_dir,
                          pred_save_dir=race_pred_save_dir)
        # by insurance
        private_train_X, government_train_X, private_train_ys, government_train_ys, _, _ = obtain_subgroups(
            task="train", pii="insurance", df=train_df)
        private_test_X, government_test_X, private_test_ys, government_test_ys, _, _ = obtain_subgroups(
            task="train", pii="insurance", df=test_df, preprocess=False)
        insurance_train_save_dir = "./data/analysis/models/ensemble/training/" + outcomes[i] + \
                                   "/ensemble-on-subgroup/insurance/"
        insurance_pred_save_dir = "./data/analysis/models/ensemble/prediction/" + outcomes[i] + \
                                  "/ensemble-on-subgroup/insurance/"
        Path(insurance_train_save_dir).mkdir(parents=True, exist_ok=True)
        Path(insurance_pred_save_dir).mkdir(parents=True, exist_ok=True)
        ensemble_top_down(estimators=y_estimators, train_Xs=[private_train_X, government_train_X],
                          train_group_ys=[private_train_ys, government_train_ys],
                          test_Xs=[private_test_X, government_test_X],
                          test_group_ys=[private_test_ys, government_test_ys],
                          group_names=["private", "government"], i=i, cols=cols,
                          train_save_dir=insurance_train_save_dir, pred_save_dir=insurance_pred_save_dir)
        # by age-group
        forties_train_X, fifties_train_X, sixties_train_X, seventies_train_X, eighty_and_over_train_X, \
            forties_train_ys, fifties_train_ys, sixties_train_ys, seventies_train_ys, \
            eighty_and_over_train_ys, _, _, _, _, _ = obtain_subgroups(task="train", pii="age-group", df=train_df)
        forties_test_X, fifties_test_X, sixties_test_X, seventies_test_X, eighty_and_over_test_X, \
            forties_test_ys, fifties_test_ys, sixties_test_ys, seventies_test_ys, \
            eighty_and_over_test_ys, _, _, _, _, _ = obtain_subgroups(task="train", pii="age-group", df=test_df,
                                                                  preprocess=False)
        age_group_train_save_dir = "./data/analysis/models/ensemble/training/" + outcomes[i] + \
                                   "/ensemble-on-subgroup/age-group/"
        age_group_pred_save_dir = "./data/analysis/models/ensemble/prediction/" + outcomes[i] + \
                                  "/ensemble-on-subgroup/age-group/"
        Path(age_group_train_save_dir).mkdir(parents=True, exist_ok=True)
        Path(age_group_pred_save_dir).mkdir(parents=True, exist_ok=True)
        ensemble_top_down(estimators=y_estimators,
                          train_Xs=[forties_train_X, fifties_train_X, sixties_train_X, seventies_train_X,
                                    eighty_and_over_train_X],
                          train_group_ys=[forties_train_ys, fifties_train_ys, sixties_train_ys, seventies_train_ys,
                                          eighty_and_over_train_ys],
                          test_Xs=[forties_test_X, fifties_test_X, sixties_test_X, seventies_test_X,
                                   eighty_and_over_test_X],
                          test_group_ys=[forties_test_ys, fifties_test_ys, sixties_test_ys, seventies_test_ys,
                                         eighty_and_over_test_ys],
                          group_names=["forties", "fifties", "sixties", "seventies", "eighty-and-over"], i=i, cols=cols,
                          train_save_dir=age_group_train_save_dir, pred_save_dir=age_group_pred_save_dir)


def plot_calibration_curve(model, X, y, save_path, save_name):
    """

    :param model:
    :param X:
    :param y:
    :param save_path:
    :param save_name:
    :return:
    """
    """code obtained from """  # todo: add url
    # predict probabilities
    probs = model.predict_proba(X)[:, 1]
    # reliability diagram
    fop, mpv = calibration_curve(y, probs, n_bins=10)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(mpv, fop, marker='.')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path + save_name)
    # plt.show()
    plt.close()


def obtain_subgroups(task, pii, df, preprocess=True):
    """

    :param task:
    :param pii:
    :param df:
    :param preprocess:
    :return:
    """
    if task == "train":  # todo: this function can be optimized to avoid repeated code
        if pii == "sex":
            male_df = df[df["gender-f"] == 0]
            female_df = df[df["gender-f"] == 1]
            male_X_cols, male_X, male_ys = process_training_data(male_df, preprocess=preprocess)
            female_X_cols, female_X, female_ys = process_training_data(female_df, preprocess=preprocess)
            return male_X, female_X, male_ys, female_ys, male_X_cols, female_X_cols
        elif pii == "insurance":
            private_df = df[df["insurance-private"] == 1]
            government_df = df[(df["insurance-medicaid"] == 1) | (df["insurance-medicare"] == 1) | (df[
                                                                                                        "insurance-government"] == 1)]
            private_X_cols, private_X, private_ys = process_training_data(private_df, preprocess=preprocess)
            government_X_cols, government_X, government_ys = process_training_data(government_df, preprocess=preprocess)
            return private_X, government_X, private_ys, government_ys, private_X_cols, government_X_cols
        elif pii == "age-group":
            forties_df = df[df["agegroup-40-49"] == 1]
            fifties_df = df[df["agegroup-50-59"] == 1]
            sixties_df = df[df["agegroup-60-69"] == 1]
            seventies_df = df[df["agegroup-70-79"] == 1]
            eighty_and_over_df = df[df["agegroup-80+"] == 1]
            forties_X_cols, forties_X, forties_ys = process_training_data(forties_df, preprocess=preprocess)
            fifties_X_cols, fifties_X, fifties_ys = process_training_data(fifties_df, preprocess=preprocess)
            sixties_X_cols, sixties_X, sixties_ys = process_training_data(sixties_df, preprocess=preprocess)
            seventies_X_cols, seventies_X, seventies_ys = process_training_data(seventies_df, preprocess=preprocess)
            eighty_and_over_X_cols, eighty_and_over_X, eighty_and_over_ys = process_training_data(eighty_and_over_df,
                                                                                                  preprocess=preprocess)
            return forties_X, fifties_X, sixties_X, seventies_X, eighty_and_over_X, forties_ys, fifties_ys, \
                   sixties_ys, seventies_ys, eighty_and_over_ys, forties_X_cols, fifties_X_cols, sixties_X_cols, \
                   seventies_X_cols, eighty_and_over_X_cols
        elif pii == "race":
            white_df = df[df["race-white/caucasian-american"] == 1]
            black_df = df[df["race-black/african-american"] == 1]
            black_latino_native_df = df[(df["race-latinx/hispanic-american"] == 1) |
                                        (df["race-black/african-american"] == 1) |
                                        (df["race-alaska-native/american-indian"] == 1)]
            non_white_df = df[(df["race-latinx/hispanic-american"] == 1) | (df["race-black/african-american"] == 1) |
                              (df["race-asian/asian-american"] == 1) | (df["race-alaska-native/american-indian"] == 1)]
            unknown_df = df[(df["race-latinx/hispanic-american"] == 0) & (df["race-black/african-american"] == 0) &
                            (df["race-asian/asian-american"] == 0) & (df["race-alaska-native/american-indian"] == 0) &
                            (df["race-white/caucasian-american"] == 1)]
            white_X_cols, white_X, white_ys = process_training_data(white_df)
            non_white_X_cols, non_white_X, non_white_ys = process_training_data(non_white_df)
            return white_X, non_white_X, white_ys, non_white_ys, white_X_cols, non_white_X_cols
    elif task == "evaluate":  # assumes segmentation is only needed for results for entire population
        if pii == "sex":
            male_df = df[df["gender-f"] == 0]
            female_df = df[df["gender-f"] == 1]
            return [male_df, female_df]
        elif pii == "insurance":
            private_df = df[df["insurance-private"] == 1]
            government_df = df[(df["insurance-medicaid"] == 1) | (df["insurance-medicare"] == 1) | df[
                "insurance-government"] == 1]
            return [private_df, government_df]
        elif pii == "age-group":
            forties_df = df[df["agegroup-40-49"] == 1]
            fifties_df = df[df["agegroup-50-59"] == 1]
            sixties_df = df[df["agegroup-60-69"] == 1]
            seventies_df = df[df["agegroup-70-79"] == 1]
            eighty_and_over_df = df[df["agegroup-80+"] == 1]
            return [forties_df, fifties_df, sixties_df, seventies_df, eighty_and_over_df]
        elif pii == "race":
            white_df = df[df["race-white/caucasian-american"] == 1]
            non_white_df = df[(df["race-latinx/hispanic-american"] == 1) | (df["race-black/african-american"] == 1) |
                              (df["race-asian/asian-american"] == 1) | (df["race-alaska-native/american-indian"] == 1)]
            return [white_df, non_white_df]


def process_training_data(train_df, test_df=None, preprocess=True):
    """

    :param train_df:
    :param test_df:
    :param preprocess:
    :return:
    """
    train_df = train_df.drop(train_df.filter(regex='Unnamed').columns, axis=1)
    train_X = train_df.drop(columns=["died-in-hosp?", "fav-disch-loc?"])
    cols = train_X.columns.tolist()
    if preprocess:  # todo: this causes a bug, when value is True, check why
        train_X = pre_modeling_pipeline(train_X=train_X)
    train_y1 = train_df["died-in-hosp?"].to_numpy()
    train_y2 = train_df["fav-disch-loc?"].to_numpy()
    if test_df is not None:
        test_df = test_df.drop(test_df.filter(regex='Unnamed').columns, axis=1)
        test_X = test_df.drop(columns=["died-in-hosp?", "fav-disch-loc?"])
        if preprocess:  # todo: this causes a bug, when value is True, check why
            train_X, test_X = pre_modeling_pipeline(train_X=train_X, test_X=test_X)
        test_y1 = test_df["died-in-hosp?"].to_numpy()
        test_y2 = test_df["fav-disch-loc?"].to_numpy()
        return cols, train_X, test_X, [train_y1, train_y2], [test_y1, test_y2]
    else:
        return cols, train_X, [train_y1, train_y2]


def evaluate_performance_of_single_models(level, pii, y_name, metric):
    _, classifier_names = classifiers_to_test()
    sex_group_names = ["male", "female"]
    race_group_names = ["white", "non-white"]
    insurance_group_names = ["private", "government"]
    age_group_names = ["forties", "fifties", "sixties", "seventies", "eighty-and-over"]
    # todo: create path to main folder here: data-training

    # entire population
    if level == "entire-population":
        groups_performance = []
        # get the classifier with the best performance
        cv_training_results_df = pd.read_csv("./data/analysis/models/single-model/training/" + y_name +
                                             "/entire-population/cv-training-results-for-" + y_name +
                                             "-entire-population.csv")
        best_ep_clf = top_model_by_metric(df=cv_training_results_df, metric=metric)
        print("best_clf = ", best_ep_clf)
        ep_predictions_df = pd.read_csv("./data/analysis/models/single-model/prediction/" + y_name +
                                        "/entire-population/features-and-prediction-results-for-" + y_name +
                                        "-entire-population.csv")
        # evaluate entire population's performance
        # print("testing====", best_ep_clf + "-preds", ep_predictions_df.columns.tolist(), ep_predictions_df[best_ep_clf+"-preds"])
        ep_f1, ep_accuracy, ep_precision, ep_recall = evaluate_model_performance(
            true_y=ep_predictions_df["true-y"], predicted_y=ep_predictions_df[best_ep_clf + "-preds"])
        groups_performance.append(["entire-population", "", ep_accuracy, ep_f1, ep_precision, ep_recall])

        # todo: add code for calibrated models

        if pii == "sex":
            groups = sex_group_names
            group_dfs = obtain_subgroups(task="evaluate", pii="sex", df=ep_predictions_df)
        elif pii == "race":
            groups = race_group_names
            group_dfs = obtain_subgroups(task="evaluate", pii="race", df=ep_predictions_df)
        elif pii == "age-group":
            groups = age_group_names
            group_dfs = obtain_subgroups(task="evaluate", pii="age-group", df=ep_predictions_df)
        elif pii == "insurance":
            groups = insurance_group_names
            group_dfs = obtain_subgroups(task="evaluate", pii="insurance", df=ep_predictions_df)
        else:
            return ValueError("The value for pii passed is incorrect. Expected values are:"
                              " 'sex', 'race', 'age-group', and 'insurance'")
        for j in range(len(group_dfs)):
            group_df = group_dfs[j]
            group_binary_f1, group_accuracy, group_precision, group_recall = evaluate_model_performance(
                true_y=group_df["true-y"], predicted_y=group_df[best_ep_clf + "-preds"])
            groups_performance.append([groups[j], pii, group_accuracy, group_binary_f1, group_precision, group_recall])
        return groups_performance
    # subpopulation
    elif level == "subgroup":
        if pii == "sex":
            groups = sex_group_names
        elif pii == "race":
            groups = race_group_names
        elif pii == "insurance":
            groups = insurance_group_names
        elif pii == "age-group":
            groups = age_group_names
        else:
            return ValueError("The value for pii passed is incorrect. Expected values are:"
                              " 'sex', 'race', 'age-group', and 'insurance'")
        # obtain models trained on each subpopulation separately
        groups_performance = []
        for j in range(len(groups)):
            group_cv_training_results_df = pd.read_csv(
                "./data/analysis/models/single-model/training/" + y_name +
                "/subgroup/" + pii + "/cv-training-results-for-"
                + y_name + "-" + groups[j] +
                "-patients.csv")
            group_best_clf = top_model_by_metric(df=group_cv_training_results_df, metric=metric)

            group_predictions_df = pd.read_csv("./data/analysis/models/single-model/prediction/" + y_name +
                                               "/subgroup/" + pii + "/features-and-prediction-results-for-"
                                               + y_name + "-" + groups[j] + "-patients.csv")
            group_binary_f1, group_accuracy, group_precision, group_recall = evaluate_model_performance(
                true_y=group_predictions_df["true-y"],
                predicted_y=group_predictions_df[group_best_clf + "-preds"])
            groups_performance.append([groups[j], pii, group_accuracy, group_binary_f1, group_precision, group_recall])
        return groups_performance

    # # todo: handle calibrated models here


def write_performance_of_models(model_type):
    y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    metrics = ["f1", "accuracy"]
    piis = ["sex", "race", "insurance", "age-group"]
    for i in range(len(y_names)):
        ep_y_performances = []
        sb_y_performances = []
        for pii in piis:
            if model_type == "single-model":
                ep_y_performances.append(evaluate_performance_of_single_models(level="entire-population", pii=pii,
                                                                               y_name=y_names[i], metric=metrics[i]))
                sb_y_performances.append(evaluate_performance_of_single_models(level="subgroup", pii=pii,
                                                                               y_name=y_names[i], metric=metrics[i]))
            elif model_type == "ensemble":
                ep_y_performances.append(evaluate_performance_of_ensemble_models(level="entire-population", pii=pii,
                                                                                 y_name=y_names[i]))
                sb_y_performances.append(evaluate_performance_of_ensemble_models(level="subgroup", pii=pii,
                                                                                 y_name=y_names[i]))
        ep_y_performances = flatten(ep_y_performances)
        sb_y_performances = flatten(sb_y_performances)
        ep_y_performances_clean = []
        [ep_y_performances_clean.append(x) for x in ep_y_performances if x not in ep_y_performances_clean]
        ep_performances_df = pd.DataFrame(data=ep_y_performances_clean, columns=["category", "pii", "accuracy", "f1",
                                                                                 "precision", "recall"])
        sb_performances_df = pd.DataFrame(data=sb_y_performances, columns=["group", "pii", "accuracy", "f1",
                                                                           "precision", "recall"])

        ep_dir = "./data/analysis/models/" + model_type + "/evaluation/" + y_names[i] + "/entire-population/"
        sb_dir = "./data/analysis/models/" + model_type + "/evaluation/" + y_names[i] + "/subgroup/"
        Path(ep_dir).mkdir(parents=True, exist_ok=True)
        Path(sb_dir).mkdir(parents=True, exist_ok=True)
        if model_type == "single-model":
            save_word = "best"
        elif model_type == "ensemble":
            save_word = "ensemble"
        else:
            return ValueError("The value for model-type passed is incorrect. Expected values are:"
                              " 'single-model' and 'ensemble'")

        print("ep_performances_df.head() = ", ep_performances_df.head())
        print("sb_performances_df.head() = ", sb_performances_df.head())
        ep_performances_df.to_csv(
            ep_dir + "performance-of-the-" + save_word + "-model-trained-on-entire-population-for-" + y_names[
                i] + ".csv")
        sb_performances_df.to_csv(
            sb_dir + "performance-of-the-" + save_word + "-model-trained-on-various-subgroups-for-" + y_names[
                i] + ".csv")


def evaluate_performance_of_ensemble_models(level, pii, y_name):
    sex_group_names = ["male", "female"]
    race_group_names = ["white", "non-white"]
    insurance_group_names = ["private", "government"]
    age_group_names = ["forties", "fifties", "sixties", "seventies", "eighty-and-over"]

    # ensembling on the entire population # bottom-up, this needs to be analyzed by computing all metrics on
    # entire pop
    if level == "entire-population":
        y_performances = []
        ep_predictions_df = pd.read_csv("./data/analysis/models/ensemble/prediction/" + y_name +
                                        "/ensemble-on-entire-population/" + pii + "/features-and-predictions-of-ensemble-by"
                                                                                  "-" + pii + ".csv")

        he_ep_f1, he_ep_accuracy, he_ep_precision, he_ep_recall = evaluate_model_performance(
            true_y=ep_predictions_df["true-y"], predicted_y=ep_predictions_df["ensemble-h-preds"])
        y_performances.append(["entire-population", pii, he_ep_accuracy, he_ep_f1, he_ep_precision, he_ep_recall])
        print("he_ep_f1 = ", he_ep_f1, "he_ep_accuracy = ", he_ep_accuracy, "he_ep_precision = ",
              he_ep_precision, "he_ep_recall = ", he_ep_recall)
        return y_performances

    elif level == "subgroup":  # todo: this code is repeated. could be moved into a separate function for reuse
        y_performances = []
        # ensembling on the subgroup
        subgroups_y_dir = "./data/analysis/models/ensemble/prediction/" + y_name + "/ensemble-on-subgroup/"
        if pii == "sex":
            groups = sex_group_names
        elif pii == "race":
            groups = race_group_names
        elif pii == "insurance":
            groups = insurance_group_names
        elif pii == "age-group":
            groups = age_group_names
        else:
            return ValueError("The value for pii passed is incorrect. Expected values are:"
                              " 'sex', 'race', 'age-group', and 'insurance'")
        # obtain models trained on each subpopulation separately
        for j in range(len(groups)):
            group_predictions_df = pd.read_csv(subgroups_y_dir + pii + "/" + groups[j] +
                                               "-features-and-predictions-of-ensemble.csv")
            group_binary_f1, group_accuracy, group_precision, group_recall = evaluate_model_performance(
                true_y=group_predictions_df["true-y"],
                predicted_y=group_predictions_df["ensemble-h-preds"])
            y_performances.append(
                [groups[j], pii, group_accuracy, group_binary_f1, group_precision, group_recall])
        return y_performances


def evaluate_model_performance(true_y, predicted_y):
    """

    :param true_y:
    :param predicted_y:
    :return:
    """
    # compute f1, recall, and precision
    # micro_f1_score = f1_score(true_y, predicted_y, average='micro')
    # macro_f1_score = f1_score(true_y, predicted_y, average='macro')
    pos_class_f1_score = f1_score(true_y, predicted_y, average='binary')
    accuracy = accuracy_score(true_y, predicted_y)
    precision = precision_score(true_y, predicted_y, average='binary')
    recall = recall_score(true_y, predicted_y, average='binary')
    return pos_class_f1_score, accuracy, precision, recall


def plot_training_results():
    """

    :return:
    """
    y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    for k in range(len(y_names)):
        results_df = pd.read_csv(
            "./data/analysis/models/" + y_names[k] + "/cv-training-results-for-" + y_names[k] + ".csv")
        mean_f1_scores = [results_df["cv-f1-mean"].tolist()]
        mean_accuracy_scores = [results_df["cv-accuracy-mean"].tolist()]
        mean_precision_scores = [results_df["cv-precision-mean"].tolist()]
        mean_recall_scores = [results_df["cv-recall-mean"].tolist()]
        mean_auc_scores = [results_df["cv-auc-mean"].tolist()]
        print("mean f1 scores = ", mean_f1_scores)
        print("mean accuracy scores = ", mean_accuracy_scores)
        save_dir = "./data/analysis/models/" + y_names[k] + "/"
        save_name = "plot-of-accuracy-f1-precision-and-recall-training-results-for-" + y_names[k]
        plot_f1_and_accuracy(f1_scores=mean_f1_scores, accuracy_scores=mean_accuracy_scores,
                             precision_scores=mean_precision_scores, recall_scores=mean_recall_scores,
                             auc_scores=mean_auc_scores,
                             xticks_labels=results_df["model-name"].unique().tolist(),
                             save_loc=save_dir + save_name)


def plot_f1_and_accuracy(f1_scores, accuracy_scores, precision_scores, recall_scores, auc_scores, xticks_labels,
                         save_loc):
    """

    :param f1_scores:
    :param accuracy_scores:
    :param precision_scores:
    :param recall_scores:
    :param auc_scores:
    :param xticks_labels:
    :param save_loc:
    :return:
    """
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)#, sharex=True)
    fig, axes = plt.subplots()  # , sharex=True)
    fig.set_size_inches(10, 7)
    fig.suptitle("Cross validated training results - k=5")
    axes.plot(accuracy_scores[0], label="accuracy", color="blue")
    # ax1.plot(accuracy_scores[1], label="all-features", color="orange")
    # ax1.set_title("Accuracy scores")
    axes.plot(f1_scores[0], label="f1", color="orange")
    # ax2.plot(f1_scores[1], label="all-features", color="orange")
    # ax2.set_title("F1 scores")
    axes.plot(precision_scores[0], label="precision", color="green")
    # ax3.plot(precision_scores[1], label="all-features", color="orange")
    # ax3.set_title("Precision scores")
    axes.plot(recall_scores[0], label="recall", color="brown")
    axes.plot(auc_scores[0], label="auc", color="red")
    # ax4.plot(recall_scores[1], label="all-features", color="orange")
    # ax4.set_title("Recall scores")
    # for ax in [ax1, ax2, ax3, ax4]:
    axes.set_xticks(np.arange(len(accuracy_scores[0])))
    axes.set_xticklabels(xticks_labels)
    # todo: start here, list values in these lists and see what min should make
    print("min = ",
          min(flatten(accuracy_scores[0] + precision_scores[0] + recall_scores[0] + auc_scores[0] + f1_scores[0])))
    # axes.set_ylim([min(flatten(accuracy_scores[0]+precision_scores[0]+recall_scores[0]+auc_scores[0]+f1_scores[0])),
    #                92])
    # fig.legend(labels=["non-sdoh-only", "all-features"], loc='upper left')
    fig.legend(labels=["accuracy", "f1", "precision", "recall", "auc"], loc='upper left')
    # plt.savefig(save_loc+".png")
    plt.show()


def plot_single_model_vs_ensemble_results(results, metric_labels, level, save_name, fig_width=12, fig_height=8):
    """

    :param results:
    :param metric_labels:
    :param level:
    :param save_name:
    :param fig_width:
    :param fig_height:
    :return:
    """

    """code obtained from - https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html"""
    x = np.arange(len(metric_labels))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(fig_width, fig_height)
    y_max = 0

    for model_type, model_results in results.items():
        print("model type = ", model_type)
        print("model results = ", model_results)
        offset = width * multiplier
        rects = ax.bar(x + offset, model_results, width, label=model_type)
        ax.bar_label(rects, padding=3)
        multiplier += 1
        y_max = max(y_max, max(model_results))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Score")
    ax.set_title("Performance of single vs ensemble model - on " + level)
    ax.set_xticks(x + width, metric_labels)
    ax.legend(loc="upper right")  # , ncols=3)
    ax.set_ylim(0, y_max + 0.25)
    plt.savefig(save_name)
    plt.show()


def plot_all_evaluation_results():
    # entire-population single model vs ensemble
    y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    for y in y_names:
        ep_y_results = {}
        ep_sm_y_results_df = pd.read_csv("./data/analysis/models/single-model/evaluation/" + y +
                                         "/entire-population/performance-of-the-best-model-trained-on-entire"
                                         "-population-for-" + y + ".csv")
        ep_sm_values = ep_sm_y_results_df[ep_sm_y_results_df["category"] == "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()[0]
        print("ep_values = ", ep_sm_values)
        ep_y_results["single-model"] = [round(x, 2) for x in ep_sm_values]

        ep_en_y_results_df = pd.read_csv("./data/analysis/models/ensemble/evaluation/" + y +
                                         "/entire-population/performance-of-the-ensemble-model-trained-on-entire"
                                         "-population-for-" + y + ".csv")
        ep_en_values = ep_en_y_results_df[["accuracy", "f1", "precision", "recall"]].values.tolist()
        ep_y_results["ensemble-by-sex"] = [round(x, 2) for x in ep_en_values[0]]
        ep_y_results["ensemble-by-race"] = [round(x, 2) for x in ep_en_values[1]]
        ep_y_results["ensemble-by-insurance"] = [round(x, 2) for x in ep_en_values[2]]
        ep_y_results["ensemble-by-age-group"] = [round(x, 2) for x in ep_en_values[3]]
        print("y_results = ", ep_y_results)
        ep_save_dir = "./data/analysis/results/entire-population/" + y + "/"
        Path(ep_save_dir).mkdir(parents=True, exist_ok=True)
        plot_single_model_vs_ensemble_results(results=ep_y_results,
                                              metric_labels=["accuracy", "f1", "precision", "recall"],
                                              level="entire-population", fig_width=11,
                                              save_name=ep_save_dir + "comparison-of-single-and-ensemble-models-on"
                                                                      "-entire-population-for-" + y + ".png")

        # subgroups
        sb_sm_y_results = pd.read_csv("./data/analysis/models/single-model/evaluation/" + y +
                                      "/subgroup/performance-of-the-best-model-trained-on-various-subgroups-for-"
                                      + y + ".csv")
        sb_en_y_results = pd.read_csv("./data/analysis/models/ensemble/evaluation/" + y +
                                      "/subgroup/performance-of-the-ensemble-model-trained-on-various-subgroups-for-"
                                      + y + ".csv")
        subgroups = sb_sm_y_results["group"].values.tolist()
        piis = sb_sm_y_results["pii"].values.tolist()
        sm_separate_results = sb_sm_y_results[["accuracy", "f1", "precision", "recall"]].values.tolist()
        en_results = sb_en_y_results[["accuracy", "f1", "precision", "recall"]].values.tolist()
        sm_shared_results = ep_sm_y_results_df[ep_sm_y_results_df["category"] != "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()
        sb_save_dir = "./data/analysis/results/subgroup/" + y + "/"
        Path(sb_save_dir).mkdir(parents=True, exist_ok=True)
        for i in range(len(subgroups)):
            sb_y_results = {}
            print("subgroup = ", subgroups[i])
            print("pii = ", piis[i])
            sb_y_results["single-model-shared"] = [round(x, 2) for x in sm_shared_results[i]]
            sb_y_results["single-model-separate"] = [round(x, 2) for x in sm_separate_results[i]]
            sb_y_results["ensemble"] = [round(x, 2) for x in en_results[i]]
            plot_single_model_vs_ensemble_results(results=sb_y_results,
                                                  metric_labels=["accuracy", "f1", "precision", "recall"],
                                                  level="subgroup-by-" + piis[i] + ": " + subgroups[i],
                                                  fig_width=9, fig_height=6,
                                                  save_name=sb_save_dir + "comparison-of-single-and-ensemble-models-on"
                                                                          "-" + subgroups[i] + "-for-" + y + ".png")


# todo: ensemble to test
#   1. ensemble on entire pop vs on subgroup - DONE
#   2. ensemble on subgroup and fit one model on entire pop - DONE
#   3. effect of including more models - PENDING
#       3.1. best subgroup + 1 other classifier vs best subgroup + multiple other classifiers - PENDING
#   4. explore when and when not to include soft voting - DONE (soft only when classifiers are well calibrated)
#   5. saving results and plotting - DONE
#   6. test on test set - IN PROGRESS
#   7. evaluate calibration


# read_data()
# predict_discharge_on_entire_population()
predict_discharge_for_sub_populations()  # todo: START here, debug this. sungroups behaving badly, could be a cols thing
# ensemble_on_entire_population()
# ensemble_on_subpopulations()
# write_performance_of_models(model_type="single-model")
# write_performance_of_models(model_type="ensemble")
# plot_all_evaluation_results()
# plot_training_results() # todo: revise this, determine if it needs to be here

# todo: plot the graph of F1 with best model, very F1 with ensemble bottom-up
