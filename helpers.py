from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


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