import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from helpers import classifiers_to_test, obtain_subgroups, flatten


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


def evaluate_performance_of_single_models(level, pii, y_name, metric):
    _, classifier_names = classifiers_to_test()
    sex_group_names = ["male", "female"]
    race_group_names = ["white", "non-white"]
    insurance_group_names = ["private", "government"]
    age_group_names = ["forties", "fifties", "sixties", "seventies", "eighty-and-over"]
    # todo: create path to main folder here: data-training

    # entire population
    if level == "entire-population":
        un_c_groups_performance = []
        c_groups_performance = []
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
        ep_f1, ep_accuracy, ep_precision, ep_recall = evaluate_model_performance(
            true_y=ep_predictions_df["true-y"], predicted_y=ep_predictions_df[best_ep_clf + "-preds"])
        un_c_groups_performance.append(["entire-population", "", ep_accuracy, ep_f1, ep_precision, ep_recall])
        ep_c_f1, ep_c_accuracy, ep_c_precision, ep_c_recall = evaluate_model_performance(
            true_y=ep_predictions_df["true-y"], predicted_y=ep_predictions_df[best_ep_clf + "-iso-c-preds"])
        c_groups_performance.append(["entire-population", "", ep_c_accuracy, ep_c_f1, ep_c_precision, ep_c_recall])

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
            group_c_binary_f1, group_c_accuracy, group_c_precision, group_c_recall = evaluate_model_performance(
                true_y=group_df["true-y"], predicted_y=group_df[best_ep_clf + "-iso-c-preds"])
            un_c_groups_performance.append([groups[j], pii, group_accuracy, group_binary_f1, group_precision,
                                            group_recall])
            c_groups_performance.append([groups[j], pii, group_c_accuracy, group_c_binary_f1, group_c_precision,
                                         group_c_recall])
        return un_c_groups_performance, c_groups_performance
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
                                        "/ensemble-on-entire-population/" + pii +
                                        "/features-and-predictions-of-ensemble-by-" + pii + ".csv")

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


def evaluate_ensemble_on_entire_population_for_subgroups(y_name, pii, subgroup):
    en_on_ep_df = pd.read_csv("./data/analysis/models/ensemble/prediction/"
                              + y_name + "/ensemble-on-entire-population/" + pii +
                              "/features-and-predictions-of-ensemble-by-"
                              + pii + ".csv")
    if pii == "sex":
        male_df, female_df = obtain_subgroups(task="evaluate", pii=pii, df=en_on_ep_df, preprocess=False)
        m_f1, m_acc, m_precision, m_recall = evaluate_model_performance(true_y=male_df["true-y"].to_numpy(),
                                                                        predicted_y=male_df[
                                                                            "ensemble-h-preds"].to_numpy())
        f_f1, f_acc, f_precision, f_recall = evaluate_model_performance(true_y=female_df["true-y"].to_numpy(),
                                                                        predicted_y=female_df[
                                                                            "ensemble-h-preds"].to_numpy())
        if subgroup == "male":
            return m_acc, m_f1, m_precision, m_recall
        elif subgroup == "female":
            return f_acc, f_f1, f_precision, f_recall
    elif pii == "race":
        white_df, non_white_df = obtain_subgroups(task="evaluate", pii=pii, df=en_on_ep_df, preprocess=False)
        w_f1, w_acc, w_precision, w_recall = evaluate_model_performance(true_y=white_df["true-y"].to_numpy(),
                                                                        predicted_y=white_df[
                                                                            "ensemble-h-preds"].to_numpy())
        nw_f1, nw_acc, nw_precision, nw_recall = evaluate_model_performance(true_y=non_white_df["true-y"].to_numpy(),
                                                                            predicted_y=non_white_df[
                                                                                "ensemble-h-preds"].to_numpy())
        if subgroup == "white":
            return w_acc, w_f1, w_precision, w_recall
        elif subgroup == "non-white":
            return nw_acc, nw_f1, nw_precision, nw_recall
    elif pii == "insurance":
        private_df, government_df = obtain_subgroups(task="evaluate", pii=pii, df=en_on_ep_df, preprocess=False)
        p_f1, p_acc, p_precision, p_recall = evaluate_model_performance(true_y=private_df["true-y"].to_numpy(),
                                                                        predicted_y=private_df[
                                                                            "ensemble-h-preds"].to_numpy())
        g_f1, g_acc, g_precision, g_recall = evaluate_model_performance(true_y=government_df["true-y"].to_numpy(),
                                                                        predicted_y=government_df[
                                                                            "ensemble-h-preds"].to_numpy())
        if subgroup == "private":
            return p_acc, p_f1, p_precision, p_recall
        elif subgroup == "government":
            return g_acc, g_f1, g_precision, g_recall
    elif pii == "age-group":
        forties_df, fifties_df, sixties_df, seventies_df, eighty_and_over_df = obtain_subgroups(
            task="evaluate", pii=pii, df=en_on_ep_df, preprocess=False)
        if subgroup == "forties":
            f1, acc, precision, recall = evaluate_model_performance(true_y=forties_df["true-y"].to_numpy(),
                                                                    predicted_y=forties_df[
                                                                        "ensemble-h-preds"].to_numpy())
            return acc, f1, precision, recall
        elif subgroup == "fifties":
            f1, acc, precision, recall = evaluate_model_performance(true_y=fifties_df["true-y"].to_numpy(),
                                                                    predicted_y=fifties_df[
                                                                        "ensemble-h-preds"].to_numpy())
            return acc, f1, precision, recall
        elif subgroup == "sixties":
            f1, acc, precision, recall = evaluate_model_performance(true_y=sixties_df["true-y"].to_numpy(),
                                                                    predicted_y=sixties_df[
                                                                        "ensemble-h-preds"].to_numpy())
            return acc, f1, precision, recall
        elif subgroup == "seventies":
            f1, acc, precision, recall = evaluate_model_performance(true_y=seventies_df["true-y"].to_numpy(),
                                                                    predicted_y=seventies_df[
                                                                        "ensemble-h-preds"].to_numpy())
            return acc, f1, precision, recall
        elif subgroup == "eighty-and-over":
            f1, acc, precision, recall = evaluate_model_performance(true_y=eighty_and_over_df["true-y"].to_numpy(),
                                                                    predicted_y=eighty_and_over_df[
                                                                        "ensemble-h-preds"].to_numpy())
            return acc, f1, precision, recall
    else:
        return ValueError("The value for pii passed is incorrect. Expected values are:"
                          " 'sex', 'race', 'age-group', and 'insurance'")


def write_performance_of_models(model_type):
    y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    metrics = ["f1", "accuracy"]
    piis = ["sex", "race", "insurance", "age-group"]
    for i in range(len(y_names)):
        ep_unc_y_performances = []
        ep_c_y_performances = []
        sb_y_performances = []
        for pii in piis:
            if model_type == "single-model":
                unc_res, c_res = evaluate_performance_of_single_models(level="entire-population", pii=pii,
                                                                       y_name=y_names[i], metric=metrics[i])
                ep_unc_y_performances.append(unc_res)
                ep_c_y_performances.append(c_res)
                sb_y_performances.append(evaluate_performance_of_single_models(level="subgroup", pii=pii,
                                                                               y_name=y_names[i], metric=metrics[i]))
            elif model_type == "ensemble":
                ep_unc_y_performances.append(evaluate_performance_of_ensemble_models(level="entire-population", pii=pii,
                                                                                     y_name=y_names[i]))
                sb_y_performances.append(evaluate_performance_of_ensemble_models(level="subgroup", pii=pii,
                                                                                 y_name=y_names[i]))
        ep_unc_y_performances = flatten(ep_unc_y_performances)
        ep_c_y_performances = flatten(ep_c_y_performances)
        sb_y_performances = flatten(sb_y_performances)
        ep_unc_y_performances_clean = []
        ep_c_y_performances_clean = []
        [ep_unc_y_performances_clean.append(x) for x in ep_unc_y_performances if x not in ep_unc_y_performances_clean]
        [ep_c_y_performances_clean.append(x) for x in ep_c_y_performances if x not in ep_c_y_performances_clean]
        ep_unc_performances_df = pd.DataFrame(data=ep_unc_y_performances_clean,
                                              columns=["category", "pii", "accuracy", "f1", "precision", "recall"])
        ep_c_performances_df = pd.DataFrame(data=ep_c_y_performances_clean,
                                            columns=["category", "pii", "accuracy", "f1", "precision", "recall"])
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

        print("ep_unc_performances_df.head() = ", ep_unc_performances_df.head())
        print("ep_c_performances_df.head() = ", ep_c_performances_df.head())
        print("sb_performances_df.head() = ", sb_performances_df.head())
        ep_unc_performances_df.to_csv(
            ep_dir + "performance-of-the-" + save_word + "-uncalibrated-model-trained-on-entire-population-for-"
            + y_names[i] + ".csv")
        ep_c_performances_df.to_csv(
            ep_dir + "performance-of-the-" + save_word + "-calibrated-model-trained-on-entire-population-for-"
            + y_names[i] + ".csv")
        sb_performances_df.to_csv(
            sb_dir + "performance-of-the-" + save_word + "-model-trained-on-various-subgroups-for-" + y_names[
                i] + ".csv")

