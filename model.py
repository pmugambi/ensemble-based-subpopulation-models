from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate

from helpers import flatten, process_training_data, obtain_subgroups, classifiers_to_test
from evaluate import obtain_best_model
from plot import plot_calibration_curve


def extract_cv_test_scores(cv_scores):
    """
    This is a helper function to obtain all the scores specified by a scoring list when models are being trained in
    cross-validation (CV) format
    :param cv_scores: the scores of all the iterations of CV
    :return: all the scores from the CV, the raw and the mean
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
    Function to train various models on the training set (X_train, train_ys) and predict on the test set
    (X_test, test_ys). This function also recalibrates the model using both sigmoid and isotonic methods,
    and if desired, calibration plots are drawn when recalibration is done on the training set using CV.
    :param X_train: training data, features only - a numpy ndarray
    :param train_ys: training data, outcomes only - two numpy vectors
    :param X_test: testing data, features only - a numpy ndarray
    :param test_ys: testing data, outcomes only - two numpy vectors
    :param X_col_names: the column names. needed for when predicted values are being saved as a dataframe
    :param y_names: the names of the outcomes. useful in specifying where to store the predicted values
    :param save_path_add_ons: additional folders to be added to the default storage location
    :param write_add_ons: additional file names to be added to the default during write
    :param plot_calibration: a binary value to check if reliability plots should be made at the point of recalibration
    on the training set. default value is False
    :param cv: the number of folds for which cross-validation will be run
    :return: nothing. output saved as csv files in "./data/analysis/models/single-model/training/", and
    "./data/analysis/models/single-model/prediction/"
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

            # fit with CV and get scores. goal is to get the classifier that's best for the outcome
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
            # predict
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
    This function runs the training and prediction of the entire training and entire test set.
    :return: Nothing. Outputs saved as csv files in the paths described in predict_discharge()
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
    This function runs the training and prediction of specific subgroups within the training and test sets.
    Patients are stratified into subgroups by sex, race, type of insurance, and age-group. First, the subgroups by
    each attribute are obtained, then predict_discharge() is called on each subgroup under that attribute
    :return: Nothing. Outputs saved as csv files in the paths described in predict_discharge()
    """
    # by sex
    train_df = pd.read_csv("./data/mimic-train.csv")
    test_df = pd.read_csv("./data/mimic-test.csv")
    piis = ["sex", "race", "insurance", "age-group"]
    for pii in piis:
        print("****** fitting a single model for each subgroup. now fitting models for: " + pii + " ****")
        if pii == "sex":
            male_train_X, female_train_X, male_train_ys, female_train_ys, _, _ = obtain_subgroups(
                task="train", pii=pii, df=train_df)
            male_test_X, female_test_X, male_test_ys, female_test_ys, male_X_cols, female_X_cols = obtain_subgroups(
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


def bottom_up_ensemble(estimators, train_X, train_y, test_X, test_y, X_columns, ensemble_cv_save_name,
                       ensemble_predictions_save_name):
    """
    This function builds one ensemble model for the training data (train_X, train_y) and evaluates it on the
    test set (text_X, test_y). Both soft (ensemble_soft) and hard (ensemble_hard) voting criteria as trained and tested
    :param estimators:
    :param train_X: the training data, features only, as a numpy ndarray
    :param train_y: the training data, outcome only, as a numpy vector
    :param test_X: the testing data, features only, as a numpy ndarray
    :param test_y: the testing data, outcome only, as a numpy vector
    :param X_columns: the names of the features to be used when the model predictions are being saved into a dataframe
    :param ensemble_cv_save_name: the filename to save the output of the training as
    :param ensemble_predictions_save_name: the filename to save the output of the prediction as
    :return: nothing. All outputs are saved as csv files.
    """
    # create our voting classifier, inputting our models
    ensemble_hard = VotingClassifier(estimators, voting="hard")
    ensemble_soft = VotingClassifier(estimators, voting="soft")

    ensemble_hard.fit(train_X, train_y)
    ensemble_soft.fit(train_X, train_y)
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

    predictions_df = pd.DataFrame(data=[test_y.tolist(), ensemble_hard_train_preds.tolist(),
                                        ensemble_soft_train_preds.tolist()])
    predictions_df = predictions_df.T
    predictions_df.columns = ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
    test_predictions_df = pd.concat([pd.DataFrame(test_X), predictions_df], axis=1)
    test_predictions_df.columns = X_columns + ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
    test_predictions_df.to_csv(ensemble_predictions_save_name)


def ensemble_on_entire_population():
    """
    This function trains and evaluates an ensemble model over the entire train/test set.
    To do that, 1) the best model for each subgroup, as generated by predict_discharge_for_sub_populations()
    is identified, and trained on the subgroup's data only. after training, it is added to a voting classifier.
    2) 2nd and 3rd best models on the entire population are identified and added to the voting classifier, untrained.
    3) all the classifiers are trained on the entire training set and evaluated on the entire test set
    :return: nothing. all outputs are written into csv files
    """
    # take best model for each group, then add them into a voting classifier,
    # then fit the classifier on entire train data
    outcomes = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    classifiers, classifiers_names = classifiers_to_test()

    # top 3 models by CV on the entire population, generated by predict_discharge_on_entire_population(),
    # are obtained and added as classifiers to build for each subgroup.
    # Because the best model is going to be included,i.e.,
    # the best classifier for some or all subgroups are going to be similar to those of the entire population,
    # it is NOT fit for the entire population. Only the 2nd and 3rd best classifiers are.
    estimators = {
        "y1-in-hosp-mortality": [
            ("rf", classifiers[classifiers_names.index("rf")]),
            ("mlp", classifiers[classifiers_names.index("mlp")])],
        "y2-favorable-discharge-loc": [
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

        if True in ["female-best" in x for x in y_estimators]:  # check if female-best model was previously
            # added (i.e., for the second outcome), and if so, remove it and add the current one.
            # If it wasn't added, first iteration, add it.
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
                                                                                    "-ensemble-by-"
                                                                                    "insurance.csv")

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
                                                                                    "-ensemble-by-"
                                                                                    "age-group.csv")


def ensemble_top_down(estimators, train_Xs, test_Xs, train_group_ys, test_group_ys, group_names, i, cols,
                      train_save_dir, pred_save_dir):
    """
    This function is similar to ensemble_bottom_up() except here, ensemble models are built for each subgroup.
    :param estimators: the classifiers to the included in the ensemble
    :param train_Xs: the training data, features only, for all subgroups
    :param test_Xs: the testing data, features only, for all subgroups
    :param train_group_ys: the training data, outcomes only, for all subgroups
    :param test_group_ys: the testing data, outcomes only, for all subgroups
    :param group_names: the names of all subgroups
    :param i: an integer value to keep track of the group being processed. # todo: this could have been done better
    :param cols: the names of the features, to be included in the created dataframe during write
    :param train_save_dir: the path to the directory where the training outputs should be saved
    :param pred_save_dir: the path to the directory where the testing outputs should be saved
    :return: nothing. all outputs are saved into a dataframe and written as csv files
    """
    for j in range(len(train_Xs)):
        ensemble_hard = VotingClassifier(estimators, voting="hard")
        ensemble_soft = VotingClassifier(estimators, voting="soft")

        ensemble_hard.fit(train_Xs[j], train_group_ys[j][i])
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
        # predict
        ensemble_hard_train_preds = ensemble_hard.predict(test_Xs[j])
        ensemble_soft_train_preds = ensemble_soft.predict(test_Xs[j])

        predictions_df = pd.DataFrame(data=[test_group_ys[j][i].tolist(), ensemble_hard_train_preds.tolist(),
                                            ensemble_soft_train_preds.tolist()])
        predictions_df = predictions_df.T
        predictions_df.columns = ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
        test_predictions_df = pd.concat([pd.DataFrame(test_Xs[j]), predictions_df], axis=1)
        test_predictions_df.columns = cols + ["true-y", "ensemble-h-preds", "ensemble-s-preds"]
        test_predictions_df.to_csv(pred_save_dir + group_names[j] + "-features-and-predictions-of-ensemble.csv")


def ensemble_on_subpopulations():
    """
    This function follows the opposite flow of ensemble_on_entire_population(). It trains and evaluates an ensemble
    model over a specific subgroup's train/test set.
    To do that, 1) the best model for the entire population, as generated by predict_discharge_for_entire_population(),
    is identified, and trained on the entire training set. After training, it is added to a voting classifier.
    2) 2nd and 3rd best models on the entire population are identified and added to the voting classifier, untrained.
    3) all the classifiers are trained on the subgroup's training set and evaluated on it's  test set
    :return: nothing. all outputs are written into csv files
    """
    outcomes = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    classifiers, classifiers_names = classifiers_to_test()

    # top 3 models by CV on the entire population, generated by predict_discharge_on_entire_population(),
    # are obtained and added as classifiers to build for each subgroup.
    # Because the best model is going to be included,i.e., the best classifier for the entire population,
    # it is NOT fit for the subgroups, to keep it from voting twice. Only the 2nd and 3rd best classifiers are.
    estimators = {
        "y1-in-hosp-mortality": [
            ("rf", classifiers[classifiers_names.index("rf")]),
            ("mlp", classifiers[classifiers_names.index("mlp")])],
        "y2-favorable-discharge-loc": [
            ("log-reg", classifiers[classifiers_names.index("log-reg")]),
            ("mlp", classifiers[classifiers_names.index("mlp")])]
    }

    train_df = pd.read_csv("./data/mimic-train.csv")
    train_df = train_df.drop(train_df.filter(regex='Unnamed').columns, axis=1)
    test_df = pd.read_csv("./data/mimic-test.csv")
    test_df = test_df.drop(test_df.filter(regex='Unnamed').columns, axis=1)
    cols, train_X, ys = process_training_data(train_df)

    for i in range(len(outcomes)):
        y_estimators = estimators[outcomes[i]]
        best_clf = obtain_best_model(level="entire-population", y_name=outcomes[i])

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
