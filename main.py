import data_preprocessing
import evaluate
import model
import plot

if __name__ == '__main__':
    """
    This function is the script that runs the project. Running main.py, will run this function, 
    which will preprocess the data, train models, evaluate them, and generate plots.
    """
    # obtain train and test sets (data_preprocessing)
    data_preprocessing.read_data()

    # train and predict (model)
    model.predict_discharge_on_entire_population()  # approaches 1 and 2
    model.predict_discharge_for_sub_populations()  # approach 3
    model.ensemble_on_entire_population()  # approach 4b
    model.ensemble_on_subpopulations()  # approach 4a

    # evaluate models (evaluate)
    evaluate.write_performance_of_models(model_type="single-model")
    evaluate.write_performance_of_models(model_type="ensemble")

    # plot results (plot)
    plot.plot_all_evaluation_results()

