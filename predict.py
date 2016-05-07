import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from util import get_bollinger_bands, get_rolling_std, compute_macd

PARAMS = {
    'days_price_changed': 2,
    'predict_time': 5,  # In days
    'volatility_tw': 10,  # In days
    'bollinger_band': 20,  # In days
}
NUM_DAYS = 5  # How many days to display from the nearest date


def run_test():
    df = pd.read_csv('HOSE/CTD.CSV', delimiter='|')

    close = df['Close'].copy()
    closed_past = close.shift(PARAMS['days_price_changed'])
    df['Open'] = (df['Open'] - closed_past)
    df['High'] = (df['High'] - closed_past)
    df['Low'] = (df['Low'] - closed_past)
    df['Volatility'] = pd.rolling_std(df['Close'], window=PARAMS['volatility_tw'])
    df['RollingMean'] = pd.rolling_mean(df['Close'], window=PARAMS['bollinger_band'])
    df['UpperBand'], df['LowerBand'] = get_bollinger_bands(df['RollingMean'], get_rolling_std(df['Close'],
                                                                                              PARAMS['bollinger_band']))
    df['MACD'], df['MACDSignal'], df['MACDHistogram'] = compute_macd(df['Close'])
    df['Close'] = (df['Close'] - closed_past)
    # df['Volume'] = (df['Volume'] - df['Volume'].shift(PARAMS['days_price_changed']))

    ignore_first_ndays = max(PARAMS['volatility_tw'], PARAMS['days_price_changed'], PARAMS['bollinger_band'])
    X = df[['Open', 'High', 'Low', 'Volume', 'Volatility', 'Close',
            'RollingMean', 'UpperBand', 'LowerBand',
            'MACD', 'MACDSignal', 'MACDHistogram']].ix[ignore_first_ndays:]
    predict_records = X.tail(NUM_DAYS).as_matrix()
    # print predict_records

    # print X
    y = (close.shift(-PARAMS['predict_time']) - close).apply(lambda x: -1 if x <= 0 else 1)
    y = y.ix[ignore_first_ndays:]

    X = X.ix[:len(y) - PARAMS['predict_time']].as_matrix()
    y = y.ix[:len(y) - PARAMS['predict_time']].as_matrix()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=44)

    # clf = LinearRegression(normalize=True)
    # clf.fit(x_train, y_train)
    # print "------------"
    # print "LinearRegression: {:.2f}%".format(clf.score(x_test, y_test) * 100)
    # print "Predict {}".format(clf.predict(predict_records))

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)
    print "------------"
    print "KNeighborsClassifier: {:.2f}%".format(clf.score(x_test, y_test) * 100)
    print "Predict {}".format(clf.predict(predict_records))

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    clf.fit(x_train, y_train)
    print "------------"
    print "RandomForest: {:.2f}%".format(clf.score(x_test, y_test) * 100)
    print "Predict {}".format(clf.predict(predict_records))

    clf = SVC()
    clf.fit(x_train, y_train)
    print "------------"
    print "SupportVectorMachine: {:.2f}%".format(clf.score(x_test, y_test) * 100)
    print "Predict {}".format(clf.predict(predict_records))

    clf = AdaBoostClassifier()
    clf.fit(x_train, y_train)
    print "------------"
    print "AdaBoostClassifier: {:.2f}%".format(clf.score(x_test, y_test) * 100)
    print "Predict {}".format(clf.predict(predict_records))

    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    print "------------"
    print "GradientBoostingClassifier: {:.2f}%".format(clf.score(x_test, y_test) * 100)
    print "Predict {}".format(clf.predict(predict_records))


if __name__ == '__main__':
    run_test()
