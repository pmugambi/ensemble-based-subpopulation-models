from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


def flatten(l):
    """
    Helper funtion to flatten a list of lists into one list.
    :param l: list of lists
    :return: a list containing all the items in all the lists
    """
    return [item for sublist in l for item in sublist]


def pre_modeling_pipeline(train_X, test_X=None):
    """
    Simple pipeline to impute missing values in the training and test datasets.
    The imputation function learned on the training set is used on the test set to avoid leakage
    :param train_X: training set features only (no labels)
    :param test_X: test set features only (no labels). This is optional to allow for CV on the training data.
    Useful during training to make sure the pipeline works well before testing on the held out test set
    :return: imputed training set (and test set, where relevant)
    """
    # impute missing values
    imputer = SimpleImputer(strategy="median")
    # imputer = SimpleImputer(strategy="mean") # in some cases, it may be worth imputing using the mean value
    train_X = imputer.fit_transform(train_X)
    if test_X is not None:
        test_X = imputer.transform(test_X)
        return train_X, test_X
    else:
        return train_X


def process_training_data(train_df, test_df=None, preprocess=True):
    """
    Function to preprocess the training data. It removes unnamed columns in the dataframe,
    extracts the features (by eliminating the outcome columns), imputes the missing values,
    and returns the training features, and outcomes, separateky as numpy arrays
    :param train_df: training dataframe
    :param test_df: test dataframe (optional). Similar to impute, this is optional to allow the modeling process to
    only use the training data until testing on the held out test is needed
    :param preprocess: check for whether to impute the feature sets
    :return: columns of the features and outcomes, training (and where applicable test) features,
    and training (and where applicable test) outcomes
    """
    train_df = train_df.drop(train_df.filter(regex='Unnamed').columns, axis=1)
    train_X = train_df.drop(columns=["died-in-hosp?", "fav-disch-loc?"])
    cols = train_X.columns.tolist()
    if preprocess:
        train_X = pre_modeling_pipeline(train_X=train_X)
    else:
        train_X = train_X.to_numpy()
    train_y1 = train_df["died-in-hosp?"].to_numpy()
    train_y2 = train_df["fav-disch-loc?"].to_numpy()
    if test_df is not None:
        test_df = test_df.drop(test_df.filter(regex='Unnamed').columns, axis=1)
        test_X = test_df.drop(columns=["died-in-hosp?", "fav-disch-loc?"])
        if preprocess:
            train_X, test_X = pre_modeling_pipeline(train_X=train_X, test_X=test_X)
        else:
            train_X = train_X.to_numpy()
            test_X = test_X.to_numpy()
        test_y1 = test_df["died-in-hosp?"].to_numpy()
        test_y2 = test_df["fav-disch-loc?"].to_numpy()
        return cols, train_X, test_X, [train_y1, train_y2], [test_y1, test_y2]
    else:
        return cols, train_X, [train_y1, train_y2]


def classifiers_to_test():
    """
    Definition of the classifiers to be tested.
    :return: various classifiers to be trained on the dataset
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


def obtain_subgroups(task, pii, df, preprocess=True):
    """
    Obtains the defined subgroups (by sex, race, insurance type, and age-group) from a dataframe
    :param task: whether the subgroup is needed for training or evaluation. Two values expected, 'train' or 'evaluate'.
    If needed the task is train, the subgroup's dataframe is pre-processed using process_training_data()
    before it is returned.
    :param pii: the personal identifier used to generate the subgroups. Four values are expected: sex, race,
    insurance type, and age-group
    :param df: the dataframe from which the subgroups should be obtained
    :param preprocess: whether the subgroup's data should be imputed or not
    :return: subgroups by the chosen identifier
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
            government_df = df[(df["insurance-medicaid"] == 1) | (df["insurance-medicare"] == 1) |
                               (df["insurance-government"] == 1)]
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
            black_df = df[df["race-black/african-american"] == 1]  # the numbers were so small,
            # these were merged along other ethnicities in non-white_df
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
    elif task == "evaluate":  # assumes segmentation is only from the entire population, i.e., df is the entire pop.
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