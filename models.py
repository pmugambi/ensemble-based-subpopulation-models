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
    dt_clf = DecisionTreeClassifier(random_state=0)
    rf_clf = RandomForestClassifier(random_state=0)
    lr_clf = LogisticRegression(random_state=0, max_iter=1500)
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=0, max_depth=1)
    mlp_clf = MLPClassifier(random_state=1, max_iter=800)
    svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=0)
    classifiers = [dt_clf, rf_clf, lr_clf, gb_clf, mlp_clf, svm_clf, xgb_clf]
    classifier_names = ["dt", "rf", "log-reg", "gb", "mlp", "svm", "xgb"]
    return classifiers, classifier_names


def predict_discharge(X, ys, X_col_names, y_names=None, save_path_add_ons=None, write_add_ons=None):
    """

    :param X:
    :param ys:
    :param X_col_names:
    :param y_names:
    :param save_path_add_ons:
    :param write_add_ons:
    :return:
    """
    # define classifiers
    if y_names is None:
        y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]

    classifiers, classifier_names = classifiers_to_test()

    # create modeling pipeline
    scoring = ["precision_macro", "recall_macro", "accuracy", "balanced_accuracy", "f1", "roc_auc"]
    for k in range(len(ys)):
        print(" working on outcome: -- ", y_names[k])
        outcome = y_names[k]
        scores_data = []
        predictions = []
        for i in range(len(classifiers)):
            y = ys[k]
            print("y = ", y, y.shape)
            model = classifiers[i]
            print("--------*testing*-------- " + classifier_names[i])

            # fit with CV and get scores
            cv_scores = cross_validate(model, X=X, y=ys[k], scoring=scoring)
            print("cv_scores = ", cv_scores)
            accuracy_scores = cv_scores["test_accuracy"]
            balanced_accuracy_scores = cv_scores["test_balanced_accuracy"]
            f1_scores = cv_scores["test_f1"]
            precision_scores = cv_scores["test_precision_macro"]
            recall_scores = cv_scores["test_recall_macro"]
            auc_scores = cv_scores["test_roc_auc"]
            scores_data.append([classifier_names[i], accuracy_scores, accuracy_scores.mean(), balanced_accuracy_scores,
                                balanced_accuracy_scores.mean(), f1_scores,
                                f1_scores.mean(), precision_scores, precision_scores.mean(),
                                recall_scores, recall_scores.mean(), auc_scores, auc_scores.mean()])

            # fit without CV
            uncalibrated_model = model.fit(X, y=ys[k])
            # calibrate models
            sigmoid_calibrated_model = CalibratedClassifierCV(model, cv=5, method='sigmoid').fit(X, ys[k])
            isotonic_calibrated_model = CalibratedClassifierCV(model, cv=5, method='isotonic').fit(X, ys[k])

            # plot and save calibration graphs for training set
            plot_calibration_curve(model=uncalibrated_model, X=X, y=ys[k],
                                   save_path="./data/calibration/reliability-plots/" + outcome + "/",
                                   save_name=classifier_names[i] + "reliability-plot.png")
            plot_calibration_curve(model=sigmoid_calibrated_model, X=X, y=ys[k],
                                   save_path="./data/calibration/reliability-plots/" + outcome + "/",
                                   save_name=classifier_names[i] + "reliability-plot.png")
            plot_calibration_curve(model=isotonic_calibrated_model, X=X, y=ys[k],
                                   save_path="./data/calibration/reliability-plots/" + outcome + "/",
                                   save_name=classifier_names[i] + "reliability-plot.png")

            # predict todo: predict on X-test
            uncalibrated_y_pred = uncalibrated_model.predict(X)
            sigmoid_calibrated_y_pred = sigmoid_calibrated_model.predict(X)
            isotonic_calibrated_y_pred = isotonic_calibrated_model.predict(X)
            predictions.append([uncalibrated_y_pred, sigmoid_calibrated_y_pred,
                                isotonic_calibrated_y_pred])
        results_df = pd.DataFrame(data=scores_data,
                                  columns=["model-name", "cv-accuracy-raw", "cv-accuracy-mean",
                                           "cv-balanced-accuracy-raw", "cv-balanced-accuracy-mean", "cv-f1-raw",
                                           "cv-f1-mean", "cv-precision-raw", "cv-precision-mean", "cv-recall-raw",
                                           "cv-recall-mean", "cv-auc-raw", "cv-auc-mean"])

        names = ["dt-preds", "dt-sig-c-preds", "dt-iso-c-preds",
                 "log-reg-preds", "log-reg-sig-c-preds", "log-reg-iso-c-preds",
                 "rf-preds", "rf-sig-c-preds", "rf-iso-c-preds",
                 "gb-preds", "gb-sig-c-preds", "gb-iso-c-preds",
                 "mlp-preds", "mlp-sig-c-preds", "mlp-iso-c-preds",
                 "svm-preds", "svm-sig-c-preds", "svm-iso-c-preds",
                 "xgb-preds", "xgb-sig-c-preds", "xgb-iso-c-preds"]
        predictions_df = pd.DataFrame(data=flatten(predictions))  # todo: check that this is working correctly
        print("predictions_df.head 1 = ", predictions_df.head())
        predictions_df = predictions_df.T
        predictions_df.columns = names
        print("predictions_df.head 2 = ", predictions_df.head())
        predictions_df["true-y"] = ys[k]
        print("predictions_df.head 3 = ", predictions_df.head())
        test_predictions_df = pd.concat([pd.DataFrame(X), predictions_df], axis=1)
        test_predictions_df.columns = X_col_names + predictions_df.columns.tolist()

        save_dir = "./data/analysis/models/" + outcome + "/"
        if save_path_add_ons is not None:
            save_dir += save_path_add_ons
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if write_add_ons is None:
            results_df.to_csv(save_dir + "cv-training-results-for-" + outcome + ".csv")
        else:
            results_df.to_csv(save_dir + "cv-training-results-for-" + outcome + write_add_ons + ".csv")

        preds_save_dir = "./data/analysis/models/predictions/" + outcome + "/"
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
    cols, train_X, ys = process_training_data(train_df)
    print("died-in-hosp% = ", np.round((ys[0].sum() / len(train_df) * 100)), ys[0].sum())
    print("fav-disch-loc% = ", np.round((ys[1].sum() / len(train_df) * 100)), ys[1].sum())
    predict_discharge(X=train_X,
                      ys=ys,
                      y_names=["y1-in-hosp-mortality", "y2-favorable-discharge-loc"],
                      X_col_names=cols,
                      save_path_add_ons="entire-population/",
                      write_add_ons="-entire-population")


def predict_discharge_for_sub_populations():
    """

    :return:
    """
    # by sex
    train_df = pd.read_csv("./data/mimic-train.csv")
    male_X, female_X, male_ys, female_ys, male_X_cols, female_X_cols = obtain_subgroups(
        task="train", pii="sex", df=train_df)

    # build models
    predict_discharge(X=male_X, ys=male_ys, X_col_names=male_X_cols,
                      save_path_add_ons="subgroups/sex/", write_add_ons="-male-patients")
    predict_discharge(X=female_X, ys=female_ys, X_col_names=female_X_cols,
                      save_path_add_ons="subgroups/sex/", write_add_ons="-female-patients")

    # by race


def obtain_best_model(level, y_name, pii=None):
    if level == "entire-population":
        cv_training_results_df = pd.read_csv(
            "./data/analysis/models/" + y_name + "/" + level + "/cv-training-results-for-"
            + y_name + "-" + level + ".csv")
        # print("cv_training_results_df = ", cv_training_results_df.head())
        best_clf = top_model_by_metric(df=cv_training_results_df)
        return best_clf
    elif level == "subgroup":
        if pii == "sex":
            male_results_df = pd.read_csv("./data/analysis/models/" + y_name + "/subgroups/sex/cv-training-results-for-"
                                          + y_name + "-male-patients.csv")
            female_results_df = pd.read_csv(
                "./data/analysis/models/" + y_name + "/subgroups/sex/cv-training-results-for-"
                + y_name + "-female-patients.csv")
            best_clf_male = top_model_by_metric(df=male_results_df)
            best_clf_female = top_model_by_metric(df=female_results_df)
            print("best male clf = ", best_clf_male)
            return best_clf_male, best_clf_female
        elif pii == "race":
            print()
    else:
        print()


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
    return best


def bottom_up_ensemble(estimators, train_X, train_y, X_columns, ensemble_cv_save_name, ensemble_predictions_save_name):
    # create our voting classifier, inputting our models
    ensemble_hard = VotingClassifier(estimators, voting="hard")
    ensemble_soft = VotingClassifier(estimators, voting="soft")

    ensemble_hard.fit(train_X, train_y)
    ensemble_soft.fit(train_X, train_y)
    print("success! getting here")
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
    print("average acc on e_y1_train = ", ensemble_soft_accuracy_scores.mean())
    print("average f1 on e_y1_train = ", ensemble_soft_f1_scores.mean())
    print("average precision on e_y1_train = ", ensemble_soft_precision_scores.mean())
    print("average recall on e_y1_train = ", ensemble_soft_recall_scores.mean())
    print("average auc on e_y1_train = ", ensemble_soft_auc_scores.mean())

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
    ensemble_hard_train_preds = ensemble_hard.predict(train_X)
    ensemble_soft_train_preds = ensemble_soft.predict(train_X)

    print("true y == ys[i] ", train_y)
    print("ensemble-hard preds = ", ensemble_hard_train_preds)
    print("ensemble-soft preds = ", ensemble_soft_train_preds)

    predictions_df = pd.DataFrame(data=[train_y.tolist(), ensemble_hard_train_preds.tolist(),
                                        ensemble_soft_train_preds.tolist()])
    predictions_df = predictions_df.T
    predictions_df.columns = ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
    test_predictions_df = pd.concat([pd.DataFrame(train_X), predictions_df], axis=1)
    test_predictions_df.columns = X_columns + ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
    print("test predictions df.head = ", test_predictions_df.head())
    test_predictions_df.to_csv(ensemble_predictions_save_name)


def ensemble_on_entire_population():
    """

    :return:
    """
    # take best model for each group, then add them into a voting classifier,
    # then fit the classifier on entire train data
    outcomes = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    classifiers, classifiers_names = classifiers_to_test()

    estimators = [("log_reg", classifiers[classifiers_names.index("log-reg")]),
                  ("xgb", classifiers[classifiers_names.index("xgb")])]

    train_df = pd.read_csv("./data/mimic-train.csv")
    train_df = train_df.drop(train_df.filter(regex='Unnamed').columns, axis=1)
    cols, train_X, ys = process_training_data(train_df)

    for i in range(len(outcomes)):
        # by sex
        male_X, female_X, male_ys, female_ys = obtain_subgroups(
            task="train", pii="sex", df=train_df)
        male_best_clf, female_best_clf = obtain_best_model(level="subgroup", y_name=outcomes[0], pii="sex")
        m_clf = classifiers[classifiers_names.index(male_best_clf)]
        f_clf = classifiers[classifiers_names.index(female_best_clf)]

        if ("female-best", f_clf.fit(female_X, female_ys[i])) not in estimators:  # todo: check if this is correct
            estimators.append(('male-best', m_clf.fit(male_X, male_ys[i])))
            estimators.append(("female-best", f_clf.fit(female_X, female_ys[i])))

        save_dir = "./data/analysis/models/ensemble/" + outcomes[i] + "/ensemble-on-entire-population/sex/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        bottom_up_ensemble(train_X=train_X, train_y=ys[i], estimators=estimators, X_columns=cols,
                           ensemble_cv_save_name=save_dir + "cv-results-for-ensemble-soft-by-sex.csv",
                           ensemble_predictions_save_name=save_dir + "features-and-predictions-of-ensemble-by-sex.csv")
        # todo: add other subgroups


def ensemble_top_down(estimators, Xs, group_ys, group_names, i, cols, save_dir):
    """

    :param estimators:
    :param Xs:
    :param group_ys:
    :param group_names:
    :param i: index of the outcome (i.e., 0: in-hospital mortality, 1: discharge location)
    :param cols:
    :return:
    """
    for j in range(len(Xs)):
        ensemble_hard = VotingClassifier(estimators, voting="hard")
        ensemble_soft = VotingClassifier(estimators, voting="soft")

        ensemble_hard.fit(Xs[j], group_ys[j][i])
        print("group = ", group_names[j], " ensemble hard fit = ", ensemble_hard)
        ensemble_soft.fit(Xs[j], group_ys[j][i])

        ensemble_soft_cv_scores = cross_validate(ensemble_soft, X=Xs[j], y=group_ys[j][i],
                                                 scoring=["precision_macro", "recall_macro", "accuracy",
                                                          "balanced_accuracy", "f1", "roc_auc"])
        ensemble_soft_accuracy_scores = ensemble_soft_cv_scores["test_accuracy"]
        ensemble_soft_balanced_accuracy_scores = ensemble_soft_cv_scores["test_balanced_accuracy"]
        ensemble_soft_f1_scores = ensemble_soft_cv_scores["test_f1"]
        ensemble_soft_precision_scores = ensemble_soft_cv_scores["test_precision_macro"]
        ensemble_soft_recall_scores = ensemble_soft_cv_scores["test_recall_macro"]
        ensemble_soft_auc_scores = ensemble_soft_cv_scores["test_roc_auc"]
        print("average acc on e_y1_train = ", ensemble_soft_accuracy_scores.mean())
        print("average f1 on e_y1_train = ", ensemble_soft_f1_scores.mean())
        print("average precision on e_y1_train = ", ensemble_soft_precision_scores.mean())
        print("average recall on e_y1_train = ", ensemble_soft_recall_scores.mean())
        print("average auc on e_y1_train = ", ensemble_soft_auc_scores.mean())

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

        ensemble_soft_results_df.to_csv(save_dir + group_names[j] + "-cv-results-for-ensemble-soft.csv")
        # # predict # todo: change this to test set
        ensemble_hard_train_preds = ensemble_hard.predict(Xs[j])
        ensemble_soft_train_preds = ensemble_soft.predict(Xs[j])

        predictions_df = pd.DataFrame(data=[group_ys[j][i].tolist(), ensemble_hard_train_preds.tolist(),
                                            ensemble_soft_train_preds.tolist()])
        predictions_df = predictions_df.T
        predictions_df.columns = ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
        test_predictions_df = pd.concat([pd.DataFrame(Xs[j]), predictions_df], axis=1)
        test_predictions_df.columns = cols + ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
        print(group_names[j] + " test predictions df.head = ", test_predictions_df.head())
        test_predictions_df.to_csv(save_dir + group_names[j] + "-features-and-predictions-of-ensemble.csv")


def ensemble_on_subpopulations():
    """

    :return:
    """
    outcomes = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    classifiers, classifiers_names = classifiers_to_test()

    estimators = [("log_reg", classifiers[classifiers_names.index("log-reg")]),
                  ("xgb", classifiers[classifiers_names.index("xgb")])]
    # estimators = [("svm", classifiers[classifiers_names.index("svm")]),
    #               ("mlp", classifiers[classifiers_names.index("mlp")])]

    train_df = pd.read_csv("./data/mimic-train.csv")
    train_df = train_df.drop(train_df.filter(regex='Unnamed').columns, axis=1)
    cols, train_X, ys = process_training_data(train_df)

    for i in range(len(outcomes)):
        best_clf = obtain_best_model(level="entire-population", y_name=outcomes[i])
        print("best_clf = ", best_clf)

        # fit best model on entire dataset
        fitted_clf = classifiers[classifiers_names.index(best_clf)].fit(train_X, ys[i])

        # add fitted clf to estimators
        if True in ["best-population-clf" in x for x in estimators]:
            estimators.remove(estimators[["best-population-clf" in x for x in estimators].index(True)])
            estimators.append(("best-population-clf", fitted_clf))
        else:
            estimators.append(("best-population-clf", fitted_clf))

        # fit sub-groups
        # by sex
        male_X, female_X, male_ys, female_ys = obtain_subgroups(task="train", pii="sex", df=train_df)
        save_dir = "./data/analysis/models/ensemble/" + outcomes[i] + "/ensemble-on-subgroups/sex/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ensemble_top_down(estimators=estimators, Xs=[male_X, female_X], group_ys=[male_ys, female_ys],
                          group_names=["male", "female"], i=i, cols=cols, save_dir=save_dir)

        # todo: add other groups


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


def obtain_subgroups(task, pii, df):
    """

    :param task:
    :param pii:
    :param df:
    :return:
    """
    if task == "train":
        if pii == "sex":
            male_df = df[df["gender-f"] == 0]
            female_df = df[df["gender-f"] == 1]
            male_X_cols, male_X, male_ys = process_training_data(male_df)
            female_X_cols, female_X, female_ys = process_training_data(female_df)
            return male_X, female_X, male_ys, female_ys, male_X_cols, female_X_cols
        elif pii == "insurance":
            print()
        elif pii == "age-group":
            print()
        elif pii == "race":
            print()
    elif task == "evaluate":  # assumes segmentation is only needed for results for entire population
        if pii == "sex":
            male_df = df[df["gender-f"] == 0]
            female_df = df[df["gender-f"] == 1]
            return male_df, female_df
        elif pii == "insurance":
            print()
        elif pii == "age-group":
            print()
        elif pii == "race":
            print()


def process_training_data(train_df):
    """

    :param train_df:
    :return:
    """
    train_df = train_df.drop(train_df.filter(regex='Unnamed').columns, axis=1)
    train_X = train_df.drop(columns=["died-in-hosp?", "fav-disch-loc?"])
    cols = train_X.columns.tolist()
    train_X = pre_modeling_pipeline(train_X=train_X)
    train_y1 = train_df["died-in-hosp?"].to_numpy()
    train_y2 = train_df["fav-disch-loc?"].to_numpy()
    return cols, train_X, [train_y1, train_y2]


def evaluate_performance_of_single_models():
    y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    metrics = ["f1", "accuracy"]
    _, classifier_names = classifiers_to_test()

    for i in range(len(y_names)):
        # by sex
        # entire population
        # get the classifier with the best performance
        cv_training_results_df = pd.read_csv("./data/analysis/models/" + y_names[i] +
                                             "/entire-population/cv-training-results-for-" + y_names[i] +
                                             "-entire-population.csv")
        best_ep_clf = top_model_by_metric(df=cv_training_results_df, metric=metrics[i])
        print("best_clf = ", best_ep_clf)
        ep_predictions_df = pd.read_csv("./data/analysis/models/predictions/" + y_names[i] +
                                        "/entire-population/features-and-prediction-results-for-" + y_names[i] +
                                        "-entire-population.csv")
        # print("predictions_df.head() = ", ep_predictions_df.head(), "predictions_df cols = ",
        #       ep_predictions_df.columns.tolist())
        # get entire population performance
        ep_f1, ep_accuracy, ep_precision, ep_recall = evaluate_model_performance(
            true_y=ep_predictions_df["true-y"], predicted_y=ep_predictions_df[best_ep_clf+"-preds"])
        print("performance of best model on entire dataset - f1:", ep_f1, " acc:", ep_accuracy,
              " precision: ", ep_precision, " recall: ", ep_recall)
        # get male and female groups
        ep_male_df, ep_female_df = obtain_subgroups(task="evaluate", pii="sex", df=ep_predictions_df)
        # print("ep_male_df.cols = ", ep_male_df.columns.tolist())

        # compute f1
        ep_male_binary_f1, ep_male_accuracy, ep_male_precision, ep_male_recall = evaluate_model_performance(
            true_y=ep_male_df["true-y"], predicted_y=ep_male_df[best_ep_clf + "-preds"])
        ep_female_binary_f1, ep_female_accuracy, ep_female_precision, ep_female_recall = evaluate_model_performance(
            true_y=ep_female_df["true-y"], predicted_y=ep_female_df[best_ep_clf + "-preds"])
        print("ep_male_binary_f1 = ", ep_male_binary_f1, "ep_male_acc = ", ep_male_accuracy,
              "ep_male_precision = ", ep_male_precision, "ep_male_recall = ", ep_male_recall)
        print("ep_female_binary_f1 = ", ep_female_binary_f1, "ep_female_acc = ", ep_female_accuracy,
              "ep_female_precision = ", ep_female_precision, "ep_female_recall = ", ep_female_recall)

        # todo: handle calibrated models here

        # by sub-groups
        sb_male_cv_training_results_df = pd.read_csv("./data/analysis/models/" + y_names[i] +
                                                     "/subgroups/sex/cv-training-results-for-" + y_names[i] +
                                                     "-male-patients.csv")
        best_sb_male_clf = top_model_by_metric(df=sb_male_cv_training_results_df, metric=metrics[i])
        sb_male_predictions_df = pd.read_csv("./data/analysis/models/predictions/" + y_names[i] +
                                             "/subgroups/sex/features-and-prediction-results-for-"
                                             + y_names[i] + "-male-patients.csv")
        # print("sb_male_predictions_df.columns.tolist() = ", sb_male_predictions_df.columns.tolist())
        sb_female_cv_training_results_df = pd.read_csv("./data/analysis/models/" + y_names[i] +
                                                       "/subgroups/sex/cv-training-results-for-" + y_names[i] +
                                                       "-female-patients.csv")
        best_sb_female_clf = top_model_by_metric(df=sb_female_cv_training_results_df, metric=metrics[i])
        sb_female_predictions_df = pd.read_csv("./data/analysis/models/predictions/" + y_names[i] +
                                               "/subgroups/sex/features-and-prediction-results-for-"
                                               + y_names[i] + "-female-patients.csv")

        sb_male_binary_f1, sb_male_accuracy, sb_male_precision, sb_male_recall = evaluate_model_performance(
            true_y=sb_male_predictions_df["true-y"], predicted_y=sb_male_predictions_df[best_sb_male_clf+"-preds"])
        sb_female_binary_f1, sb_female_accuracy, sb_female_precision, sb_female_recall = evaluate_model_performance(
            true_y=sb_female_predictions_df["true-y"], predicted_y=sb_female_predictions_df[best_sb_female_clf+"-preds"])
        print("sb_male_binary_f1 = ", sb_male_binary_f1, "sb_male_acc = ", sb_male_accuracy,
              "sb_male_precision = ", sb_male_precision, "sb_male_recall = ", sb_male_recall)
        print("sb_female_binary_f1 = ", sb_female_binary_f1, "sb_female_acc = ", sb_female_accuracy,
              "sb_female_precision = ", sb_female_precision, "sb_female_recall = ", sb_female_recall)

        # todo: handle calibrated models here

        # todo: extract the bulk of this code and make a function on the side, to be reused by the other groups


def evaluate_performance_of_ensemble_models():
    y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    metrics = ["f1", "accuracy"]
    _, classifier_names = classifiers_to_test()

    for i in range(len(y_names)):
        # ensembling on the entire population # bottom-up, this needs to be analyzed by computing all metrics on
        # entire pop
        ep_predictions_df = pd.read_csv("./data/analysis/models/ensemble/"+y_names[i]+
                                        "/ensemble-on-entire-population/sex/features-and-predictions-of-ensemble-by"
                                        "-sex.csv")

        he_ep_f1, he_ep_accuracy, he_ep_precision, he_ep_recall = evaluate_model_performance(
            true_y=ep_predictions_df["true-y"], predicted_y=ep_predictions_df["ensemble-h-preds"])
        se_ep_f1, se_ep_accuracy, se_ep_precision, se_ep_recall = evaluate_model_performance(
            true_y=ep_predictions_df["true-y"], predicted_y=ep_predictions_df["ensemble-s-preds"])
        # true - y, ensemble - h - preds, ensemble - s - preds
        print("he_ep_f1 = ", he_ep_f1, "he_ep_accuracy = ", he_ep_accuracy, "he_ep_precision = ",
              he_ep_precision, "he_ep_recall = ", he_ep_recall)
        print("se_ep_f1 = ", se_ep_f1, "se_ep_accuracy = ", se_ep_accuracy, "se_ep_precision = ",
              se_ep_precision, "se_ep_recall = ", se_ep_recall)

        # ensembling on the subgroups
        # todo: START HERE


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


# todo: write code to fit the various models
# 1. one model for all - DONE
# 2. one model with calibration - DONE
# 3. group-specific models - DONE
# 4. DP (double prioritized) with raw data - SKIP?*
# 5. DP with synthetic data - SKIP?*
# 6. ensemble - DONE


# todo: ensemble to test
#   1. ensemble on entire pop vs on subgroup
#   2. ensemble on subgroup and fit one model on entire pop
#   3. effect of including more models
#       3.1. best subgroups + 1 other classifier vs best subgroups + multiple other classifiers
#   4. explore when and when not to include soft voting
#   5. saving results and plotting


# read_data()
# predict_discharge_on_entire_population()
# predict_discharge_for_sub_populations()
# ensemble_on_entire_population()
# ensemble_on_subpopulations()
evaluate_performance_of_single_models()
evaluate_performance_of_ensemble_models()
# plot_training_results()

# todo: plot the graph of F1 with best model, very F1 with ensemble bottom-up