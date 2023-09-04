import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from sklearn.calibration import calibration_curve
from evaluate import evaluate_ensemble_on_entire_population_for_subgroups


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


def plot_single_model_vs_ensemble_results(results, metric_labels, level, save_name, fig_width=12, fig_height=8,
                                          colors=None, comparisons_string="single vs ensemble model"):
    """

    :param results:
    :param metric_labels:
    :param level:
    :param save_name:
    :param fig_width:
    :param fig_height:
    :param colors:
    :param comparisons_string:
    :return:
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
        # print("model type = ", model_type)
        # print("model results = ", model_results)
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
    ax.set_ylabel("Score")
    ax.set_title("Performance of " + comparisons_string + " - on " + level)
    ax.set_xticks(x + width, metric_labels)
    ax.legend(loc="upper right")  # , ncols=3)
    ax.set_ylim(0, y_max + 0.25)
    plt.savefig(save_name)
    # plt.show()
    plt.close()


def plot_all_evaluation_results():
    # entire-population single model vs ensemble
    y_names = ["y1-in-hosp-mortality", "y2-favorable-discharge-loc"]
    for y in y_names:
        ep_y_results = {}
        ep_unc_vs_c_results = {}
        ep_sm_unc_y_results_df = pd.read_csv("./data/analysis/models/single-model/evaluation/" + y +
                                             "/entire-population/performance-of-the-best-uncalibrated-model-trained-on"
                                             "-entire-population-for-" + y + ".csv")
        ep_sm_c_y_results_df = pd.read_csv("./data/analysis/models/single-model/evaluation/" + y +
                                           "/entire-population/performance-of-the-best-calibrated-model-trained-on"
                                           "-entire-population-for-" + y + ".csv")

        # build and plot models of entire population, single-uncalibrated vs single-calibrated
        ep_unc_values = ep_sm_unc_y_results_df[ep_sm_unc_y_results_df["category"] == "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()[0]
        ep_c_values = ep_sm_c_y_results_df[ep_sm_c_y_results_df["category"] == "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()[0]

        print("ep_unc_values = ", ep_unc_values)
        print("ep_cc_values = ", ep_c_values)
        ep_unc_vs_c_results["uncalibrated"] = [round(x, 2) for x in ep_unc_values]
        ep_unc_vs_c_results["calibrated"] = [round(x, 2) for x in ep_c_values]
        ep_save_dir = "./data/analysis/results/entire-population/" + y + "/"
        Path(ep_save_dir).mkdir(parents=True, exist_ok=True)
        plot_single_model_vs_ensemble_results(results=ep_unc_vs_c_results,
                                              metric_labels=["accuracy", "f1", "precision", "recall"],
                                              level="entire-population", fig_width=8, fig_height=6,
                                              save_name=ep_save_dir + "comparison-of-uncalibrated-and-calibrated-one"
                                                                      "-model-for-all-fit-on-entire-population-for-"
                                                        + y + ".png",
                                              comparisons_string="single uncalibrated vs calibrated model")

        # build and plot models of entire population, single-uncalibrated vs ensemble
        ep_sm_values = ep_sm_unc_y_results_df[ep_sm_unc_y_results_df["category"] == "entire-population"][
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
        sm_unc_shared_results = ep_sm_unc_y_results_df[ep_sm_unc_y_results_df["category"] != "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()
        sm_c_shared_results = ep_sm_c_y_results_df[ep_sm_c_y_results_df["category"] != "entire-population"][
            ["accuracy", "f1", "precision", "recall"]].values.tolist()
        sb_save_dir = "./data/analysis/results/subgroup/" + y + "/"
        Path(sb_save_dir).mkdir(parents=True, exist_ok=True)
        for i in range(len(subgroups)):
            sb_en_ep_results = evaluate_ensemble_on_entire_population_for_subgroups(y_name=y, pii=piis[i],
                                                                                    subgroup=subgroups[i])
            sb_y_results = {}
            sb_unc_vs_c_results = {}
            print("subgroup = ", subgroups[i])
            print("pii = ", piis[i])
            sb_y_results["one-model-for-all"] = [round(x, 2) for x in sm_unc_shared_results[i]]
            sb_y_results["one-model-per-subgroup"] = [round(x, 2) for x in sm_separate_results[i]]
            sb_y_results["ensemble-on-all"] = [round(x, 2) for x in sb_en_ep_results]
            sb_y_results["ensemble-on-subgroup"] = [round(x, 2) for x in en_results[i]]
            plot_single_model_vs_ensemble_results(results=sb_y_results,
                                                  metric_labels=["accuracy", "f1", "precision", "recall"],
                                                  level="subgroup-by-" + piis[i] + ": " + subgroups[i],
                                                  fig_width=10, fig_height=7,
                                                  save_name=sb_save_dir + "comparison-of-single-and-ensemble-models-on"
                                                                          "-" + subgroups[i] + "-for-" + y + ".png",
                                                  colors=["cornflowerblue", "blue", "sandybrown", "sienna"])

            # calibrated vs uncalibrated models for all, performance per subgroup
            sb_calibration_save_dir = sb_save_dir + "/calibration/"
            Path(sb_calibration_save_dir).mkdir(parents=True, exist_ok=True)
            sb_unc_vs_c_results["uncalibrated-one-model-for-all"] = [round(x, 2) for x in sm_unc_shared_results[i]]
            sb_unc_vs_c_results["calibrated-one-model-for-all"] = [round(x, 2) for x in sm_c_shared_results[i]]
            plot_single_model_vs_ensemble_results(results=sb_unc_vs_c_results,
                                                  metric_labels=["accuracy", "f1", "precision", "recall"],
                                                  level="subgroup-by-" + piis[i] + ": " + subgroups[i],
                                                  fig_width=8, fig_height=6,
                                                  save_name=sb_calibration_save_dir +
                                                            "comparison-of-single-calibrated-vs-uncalibrated-model-for-"
                                                            + y + "-for-" + subgroups[i] + ".png",
                                                  comparisons_string="single uncalibrated vs calibrated model")

    # colors = ["cornflowerblue", "royalblue", "orange", "darkorange"]
    # colors = ["cornflowerblue", "blue", "moccasin", "orange"]
