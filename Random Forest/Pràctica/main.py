import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sklearn.datasets
from random_forest import RandomForestClassifier, RandomForestRegressor, \
    IsolationForest
from impurity_measure import Gini, Entropy, SumSquareError

if __name__ == '__main__':

    test = 'iris'
    X = None
    y = None

    if test == 'iris' or test == 'sonar':

        if test == 'iris':
            iris = sklearn.datasets.load_iris()
            X, y = iris.data, iris.target
            # X 150 x 4, y 150 numpy arrays

        if test == 'sonar':
            def load_sonar():
                df = pd.read_csv('C:/Users/helen/OneDrive/Documentos/1r Carrera'
                                 '/2n semestre/Programació orientada als '
                                 'objectes/Pràctica/sonar.all-data',
                                 header=None)
                samples = df[df.columns[:-1]].to_numpy()
                features = df[df.columns[-1]].to_numpy(dtype=str)
                features = (features == 'M').astype(int)  # M = mine, R = rock
                return samples, features


            X, y = load_sonar()

        ratio_train, ratio_test = 0.7, 0.3  # percentatge de mostres de cada
        # fase, 70% train, 30% test
        num_samples, num_features = X.shape  # samples són files i features
        # columnes( 150, 4)
        idx = np.random.permutation(range(num_samples))  # shuffle
        # {0,1, ... 149} because samples come sorted by class!

        num_samples_train = int(num_samples * ratio_train)  # Nombre de mostres
        # de cada fase
        num_samples_test = int(num_samples * ratio_test)
        idx_train = idx[:num_samples_train]  # Guarda el num de les mostres ja
        # desordenades
        idx_test = idx[num_samples_train: num_samples_train + num_samples_test]

        X_train, y_train = X[idx_train], y[idx_train]  # Guarda les mostres
        X_test, y_test = X[idx_test], y[idx_test]

        max_depth = 10  # maximum number of levels of a decision tree
        min_size = 5  # if less, do not split a node
        ratio_samples = 0.7  # sampling with replacement
        num_trees = 20  # number of decision trees
        num_random_features = int(np.sqrt(num_features))
        impurity_measure = Entropy()

        rf = RandomForestClassifier(max_depth, min_size, ratio_samples,
                                    num_trees, num_random_features,
                                    impurity_measure)

        rf.fit(X_train, y_train)  # train = make the decision trees
        ypred = rf.predict(X_test)  # classification

        num_samples_test = len(y_test)
        num_correct_predictions = np.sum(ypred == y_test)
        accuracy = num_correct_predictions / float(num_samples_test)
        print('accuracy {} %'.format(100 * np.round(accuracy, decimals=2)))

        if test == 'iris':

            occurrences = rf.feature_importance()
            print('Iris occurrences for {} trees'.format(rf.num_trees))
            print(occurrences)

        elif test == 'sonar':

            occurrences = rf.feature_importance()  # a dictionary
            counts = np.array(list(occurrences.items()))
            plt.figure(), plt.bar(counts[:, 0], counts[:, 1])
            plt.xlabel('feature')
            plt.ylabel('occurrences')
            plt.title('Sonar feature importance\n{} trees'.format(rf.num_trees))
            plt.show()

    elif test == 'mnist':

        def load_MNIST():
            with open("mnist.pkl", 'rb') as f:
                mnist = pickle.load(f)
            return mnist["training_images"], mnist["training_labels"], \
                mnist["test_images"], mnist["test_labels"]


        X_train, y_train, X_test, y_test = load_MNIST()

        max_depth = 20
        min_size = 20
        ratio_samples = 0.25
        num_random_features = 28  # int(np.sqrt(X_train.shape[1]))
        num_trees = 80
        impurity_measure = Gini()

        rf = RandomForestClassifier(max_depth, min_size, ratio_samples,
                                    num_trees, num_random_features,
                                    impurity_measure)

        rf.fit(X_train, y_train)  # train = make the decision trees
        ypred = rf.predict(X_test)  # classification

        num_samples_test = len(y_test)
        num_correct_predictions = np.sum(ypred == y_test)
        accuracy = num_correct_predictions / float(num_samples_test)
        print('accuracy {} %'.format(100 * np.round(accuracy, decimals=2)))

        occurrences = rf.feature_importance()
        ima = np.zeros(28 * 28)
        for k in occurrences.keys():
            ima[k] = occurrences[k]
        plt.figure()
        plt.imshow(np.reshape(ima, (28, 28)))
        plt.colorbar()
        plt.title('Feature importance MNIST')
        plt.show()

    elif test == 'temperatures':

        def load_daily_min_temperatures():
            df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/'
                             'Datasets/master/daily-min-temperatures.csv')
            day = pd.DatetimeIndex(df.Date).day.to_numpy()  # 1...31
            month = pd.DatetimeIndex(df.Date).month.to_numpy()  # 1...12
            year = pd.DatetimeIndex(df.Date).year.to_numpy()  # 1981...1999
            samples = np.vstack([day, month, year]).T  # np array of 3 columns
            features = df.Temp.to_numpy()
            return samples, features


        X, y = load_daily_min_temperatures()
        last_years_test = 1

        plt.figure()
        plt.plot(y, '.-')
        plt.xlabel('day in 10 years'), plt.ylabel('min. daily temperature')
        idx = last_years_test * 365
        X_train = X[:-idx, :]  # first years
        X_test = X[-idx:]
        y_train = y[:-idx]
        y_test = y[-idx:]

        max_depth = 10  # maximum number of levels of a decision tree
        min_size = 5  # if less, do not split a node
        ratio_samples = 0.5  # sampling with replacement
        num_trees = 50  # number of decision trees
        num_random_features = 2
        impurity_measure = SumSquareError()

        rf = RandomForestRegressor(max_depth, min_size, ratio_samples,
                                   num_trees, num_random_features,
                                   impurity_measure)

        rf.fit(X_train, y_train)  # train = make the decision trees
        ypred = rf.predict(X_test)  # classification

        plt.figure()
        x = range(idx)
        for t, y1, y2 in zip(x, y_test, ypred):
            plt.plot([t, t], [y1, y2], 'k-')
        plt.plot([x[0], x[0]], [y_test[0], ypred[0]], 'k-', label='error')
        plt.plot(x, y_test, 'g.', label='test')
        plt.plot(x, ypred, 'y.', label='prediction')
        plt.xlabel('day in last {} years'.format(last_years_test))
        plt.ylabel('min. daily temperature')
        plt.legend()
        errors = y_test - ypred
        rmse = np.sqrt(np.mean(errors ** 2))
        plt.title('root mean square error : {:.3f}'.format(rmse))
        plt.show()

    elif test == 'isolation':

        rng = np.random.RandomState(42)
        X = 0.3 * rng.randn(100, 2)  # synthetic dataset, two Gaussians
        X_train = np.r_[X + 2, X - 2]
        X = 0.3 * rng.randn(20, 2)
        X_test = np.r_[X + 2, X - 2]
        X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
        xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        Xgrid = np.c_[xx.ravel(), yy.ravel()]  # where to compute the score

        iso = IsolationForest(num_trees=100, ratio_samples=0.5)
        iso.fit(X_train)
        scores = iso.predict(Xgrid)

        scores = scores.reshape(xx.shape)
        plt.title("IsolationForest")
        m = plt.contourf(xx, yy, scores, cmap=plt.cm.Blues_r)
        plt.colorbar(m)
        b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=20,
                         edgecolor="k")
        b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="green", s=20,
                         edgecolor="k")
        c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="red", s=20,
                        edgecolor="k")
        plt.axis("tight"), plt.xlim((-5, 5)), plt.ylim((-5, 5))
        plt.legend([b1, b2, c], ["training observations",
                                 "new regular observations",
                                 "new abnormal observations"], loc="upper left")
        plt.show(block=False)

    elif test == 'credit_card':

        def test_credit_card_fraud():
            df = pd.read_csv('C:/Users/helen/OneDrive/Documentos/1r Carrera'
                             '/2n semestre/Programació orientada als '
                             'objectes/Pràctica/creditcard_10K.csv',
                             header=None)

            samples = np.array(df)
            samples = samples[:, 1:]  # remove first feature
            features = samples[:, -1]
            samples = samples[:, :-1]
            del df
            num_samples1 = len(samples)
            print('{} number of samples'.format(num_samples1))
            np.random.seed(123)  # to get replicable results
            idx1 = np.random.permutation(num_samples1)
            samples = samples[idx1]  # shuffle
            features = features[idx1]
            print('{} samples, {} outliers, {} % '.
                  format(len(features), features.sum(),
                         np.round(100 * features.sum() / len(features),
                                  decimals=3)))
            num_trees1 = 500
            ratio_samples1 = 0.1
            iso1 = IsolationForest(num_trees1, ratio_samples1)

            iso1.fit(samples)
            scores1 = iso1.predict(samples)

            plt.figure(), plt.hist(scores1, bins=100)
            plt.title('histogram of scores')

            percent_anomalies = 0.5
            num_anomalies = int(percent_anomalies * num_samples1 / 100.)
            idx1 = np.argsort(scores1)
            idx_predicted_anomalies = idx1[-num_anomalies:]
            precision = features[idx_predicted_anomalies].sum() / num_anomalies
            print('precision for {} % anomalies : {} %'
                  .format(percent_anomalies, 100 * precision))
            recall = features[idx_predicted_anomalies].sum() / features.sum()
            print('recall for {} % anomalies : {} %'
                  .format(percent_anomalies, 100 * recall))


        test_credit_card_fraud()

    elif test == 'mnist1':

        def test_MNIST(digit=8):

            def load_MNIST1():
                with open("mnist.pkl", 'rb') as f:
                    mnist = pickle.load(f)
                return mnist["training_images"], mnist["training_labels"], \
                    mnist["test_images"], mnist["test_labels"]

            X_train1, y_train1, X_test1, y_test1 = load_MNIST1()
            X2 = np.vstack([X_train1, X_test1])
            num_samples1 = len(X2)
            y3 = np.concatenate([y_train1, y_test1])
            idx_digit = np.where(y3 == digit)[0]
            X2 = X2[idx_digit]
            downsample = 2
            X3 = np.reshape(X2,
                            (len(X2), 28, 28))[:, ::downsample, ::downsample]
            X3 = np.reshape(X3, (len(X3), 28 * 28 // downsample ** 2))

            iso2 = IsolationForest(num_trees=2000, ratio_samples=0.002)
            iso2.fit(X3)
            scores1 = iso2.predict(X3)

            percent_anomalies = 0.2
            num_anomalies = int(percent_anomalies * num_samples1 / 100.)
            idx_digit = np.argsort(scores1)
            idx_predicted_anomalies = idx_digit[-num_anomalies:]
            recall = y3[idx_predicted_anomalies].sum() / y3.sum()
            print('recall for {} % anomalies : {} %'
                  .format(percent_anomalies, 100 * recall))

        test_MNIST(digit=4)
