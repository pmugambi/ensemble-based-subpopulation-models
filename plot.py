import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from sklearn.calibration import calibration_curve
from evaluate import evaluate_ensemble_on_entire_population_for_subgroups


def plot_calibration_curve(model, X, y, save_path, save_name):
    """

    :param model: the model specification
    :param X: data features
    :param y: data outcomes/labels
    :param save_path: the path where the plots should be saved
    :param save_name: the filename the plot should be saved under
    :return: nothing. plots and saves the reliability plots
    """
    """code obtained from https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/"""
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


def plot_single_model_vs_ensemble_results(results, metric_labels, level, save_name, outcome, fig_width=12, fig_height=8,
                                          colors=None, comparisons_string="single vs ensemble model"):
    """

    :param results: a dictionary of the model's performance.
    :param metric_labels: the labels of the metrics used to evaluate the model's performance. e.g., "F1", "Accuracy"
    :param level: level at which the model is being evaluated. either population or subgroup
    :param save_name: the name under which the file should be saved
    :param outcome: the outcome for which the results were generated.
                    Takes two values: "in-hospital mortality" or "discharge location"
    :param fig_width: the width of the figure to the generated
    :param fig_height: the height of the figure to the generated
    :param colors: the colors for the respective bars. By default, Matplotlib defaults are used i.e., color = None
    :param comparisons_string: string to be appended in the title and file name indicating what is being compared
    :return: nothing. generated plots are saved in ./data/analysis/results
    """

    """code obtained from - https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html"""
    x = np.arange(len(metric_labels))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(fig_width, fig_height)
    y_max = 0
    ind = 0

    for model_type, model_results in results.items():
        offset = width * multiplier
        if colors is not None:
            rects = ax.bar(x + offset, model_results, width, label=model_type, color=colors[ind])
        else:
            rects = ax.bar(x + offset, model_results, width, label=model_type)
        ax.bar_label(rects, padding=3)
        multiplier += 1
        y_max = max(y_max, max(model_results))
        ind += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Score(%)", fontsize=12)
    ax.set_title(outcome + ": performance of " + comparisons_string + " on " + level)
    ax.set_xticks(x + width, metric_labels, fontsize=12)
    ax.legend(loc="upper right")  # , ncols=3)
    ax.set_ylim(0, y_max + 22)
    plt.savefig(save_name)
    # plt.show()
    plt.close()


def plot_all_evaluation_results():
    """
    Function to generate all plots in the project
    :return:
    """
    # 1. entire-population
    y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    for j in range(len(y_names)):
        y = y_names[j]
        ep_y_results = {}
        ep_unc_vs_c_results = {}
        ep_sm_unc_y_results_df = pd.read_csv("./data/analysis/models/single-model/evaluation/" + y +
                                             "/entire-population/performance-of-the-best-uncalibrated-model-trained-on"
                                             "-entire-population-for-" + y + ".csv")
        ep_sm_c_y_results_df = pd.read_csv("./data/analysis/models/single-model/evaluation/" + y +
                                           "/entire-population/performance-of-the-best-calibrated-model-trained-on"
                                           "-entire-population-for-" + y + ".csv")

        # 1.1 build and plot models of entire population, single-uncalibrated vs single-calibrated
        ep_unc_values = ep_sm_unc_y_results_df[ep_sm_unc_y_results_df["category"] == "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()[0]
        ep_c_values = ep_sm_c_y_results_df[ep_sm_c_y_results_df["category"] == "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()[0]

        print("ep_unc_values = ", ep_unc_values)
        print("ep_cc_values = ", ep_c_values)
        ep_unc_vs_c_results["uncalibrated"] = [round(x * 100, 1) for x in ep_unc_values]
        ep_unc_vs_c_results["calibrated"] = [round(x * 100, 1) for x in ep_c_values]
        ep_save_dir = "./data/analysis/results/entire-population/" + y + "/"
        Path(ep_save_dir).mkdir(parents=True, exist_ok=True)
        plot_single_model_vs_ensemble_results(results=ep_unc_vs_c_results,
                                              metric_labels=["accuracy", "f1", "precision", "recall"],
                                              level="entire-population", fig_width=8, fig_height=6,
                                              outcome="Y" + str(j + 1),
                                              save_name=ep_save_dir + "comparison-of-uncalibrated-and-calibrated-one"
                                                                      "-model-for-all-fit-on-entire-population-for-"
                                                        + y + ".pdf",
                                              comparisons_string="single uncalibrated vs calibrated model")

        # 1.2 build and plot models of entire population, single-uncalibrated vs ensemble
        ep_sm_values = ep_sm_unc_y_results_df[ep_sm_unc_y_results_df["category"] == "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()[0]
        print("ep_values = ", ep_sm_values)
        ep_y_results["single-model"] = [round(x * 100, 1) for x in ep_sm_values]

        ep_en_y_results_df = pd.read_csv("./data/analysis/models/ensemble/evaluation/" + y +
                                         "/entire-population/performance-of-the-ensemble-model-trained-on-entire"
                                         "-population-for-" + y + ".csv")
        ep_en_values = ep_en_y_results_df[["accuracy", "f1", "precision", "recall"]].values.tolist()
        ep_y_results["ensemble-by-sex"] = [round(x * 100, 1) for x in ep_en_values[0]]
        ep_y_results["ensemble-by-race"] = [round(x * 100, 1) for x in ep_en_values[1]]
        ep_y_results["ensemble-by-insurance"] = [round(x * 100, 1) for x in ep_en_values[2]]
        ep_y_results["ensemble-by-age-group"] = [round(x * 100, 1) for x in ep_en_values[3]]
        print("y_results = ", ep_y_results)

        plot_single_model_vs_ensemble_results(results=ep_y_results,
                                              metric_labels=["accuracy", "f1", "precision", "recall"],
                                              level="entire-population", fig_width=11, outcome="Y" + str(j + 1),
                                              save_name=ep_save_dir + "comparison-of-single-and-ensemble-models-on"
                                                                      "-entire-population-for-" + y + ".pdf")

        # 2. subgroups
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
        sm_unc_shared_results = ep_sm_unc_y_results_df[ep_sm_unc_y_results_df["category"] != "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()
        sm_c_shared_results = ep_sm_c_y_results_df[ep_sm_c_y_results_df["category"] != "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()
        sb_save_dir = "./data/analysis/results/subgroup/" + y + "/"
        Path(sb_save_dir).mkdir(parents=True, exist_ok=True)
        for i in range(len(subgroups)):
            # 2.1 uncalibrated single model for each subgroup vs ensemble by subgroup model
            sb_en_ep_results = evaluate_ensemble_on_entire_population_for_subgroups(y_name=y, pii=piis[i],
                                                                                    subgroup=subgroups[i])
            sb_y_results = {}
            sb_unc_vs_c_results = {}
            print("subgroup = ", subgroups[i])
            print("pii = ", piis[i])
            sb_y_results["one-model-for-all"] = [round(x * 100, 1) for x in sm_unc_shared_results[i]]
            sb_y_results["one-model-per-subgroup"] = [round(x * 100, 1) for x in sm_separate_results[i]]
            sb_y_results["ensemble-on-all"] = [round(x * 100, 1) for x in sb_en_ep_results]
            sb_y_results["ensemble-on-subgroup"] = [round(x * 100, 1) for x in en_results[i]]
            plot_single_model_vs_ensemble_results(results=sb_y_results,
                                                  metric_labels=["accuracy", "f1", "precision", "recall"],
                                                  level="subgroup-by-" + piis[i] + ": " + subgroups[i],
                                                  fig_width=10, fig_height=7, outcome="Y" + str(j + 1),
                                                  save_name=sb_save_dir + "comparison-of-single-and-ensemble-models-on"
                                                                          "-" + subgroups[i] + "-for-" + y + ".pdf",
                                                  colors=["cornflowerblue", "blue", "sandybrown", "sienna"])

            # 2.2. calibrated vs uncalibrated models for all, performance per subgroup
            sb_calibration_save_dir = sb_save_dir + "/calibration/"
            Path(sb_calibration_save_dir).mkdir(parents=True, exist_ok=True)
            sb_unc_vs_c_results["uncalibrated-one-model-for-all"] = [round(x * 100, 1) for x in
                                                                     sm_unc_shared_results[i]]
            sb_unc_vs_c_results["calibrated-one-model-for-all"] = [round(x * 100, 1) for x in sm_c_shared_results[i]]
            plot_single_model_vs_ensemble_results(results=sb_unc_vs_c_results,
                                                  metric_labels=["accuracy", "f1", "precision", "recall"],
                                                  level="subgroup-by-" + piis[i] + ": " + subgroups[i],
                                                  fig_width=10, fig_height=6, outcome="Y" + str(j + 1),
                                                  save_name=sb_calibration_save_dir +
                                                            "comparison-of-single-calibrated-vs-uncalibrated-model-for-"
                                                            + y + "-for-" + subgroups[i] + ".pdf",
                                                  comparisons_string="single uncalibrated vs calibrated model")

    # colors = ["cornflowerblue", "royalblue", "orange", "darkorange"]
    # colors = ["cornflowerblue", "blue", "moccasin", "orange"]
