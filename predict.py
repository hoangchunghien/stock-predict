import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
import util
from util import get_bollinger_bands, get_rolling_std, compute_macd

PARAMS = {
    'days_price_changed': 2,
    'predict_time': 5,
    'volatility_tw': 15,
    'bollinger_band': 20,
    'stochastic_oscillator': 14,
}
LAST_N_DAYS = 5


def prepare_train_data(symbol):
    df = pd.read_csv('HOSE/{}.CSV'.format(symbol), delimiter='|')

    close = df['Close'].copy()
    closed_past = close.shift(PARAMS['days_price_changed'])
    df['Open'] = (df['Open'] - closed_past)
    df['High'] = (df['High'] - closed_past)
    df['Low'] = (df['Low'] - closed_past)
    df['Volatility'] = df['Close'].rolling(window=PARAMS['volatility_tw']).std()
    df['RollingMean'] = df['Close'].rolling(window=PARAMS['bollinger_band']).mean()
    df['UpperBand'], df['LowerBand'] = get_bollinger_bands(df['RollingMean'], get_rolling_std(df['Close'],
                                                                                              PARAMS['bollinger_band']))

    df['MACD'], df['MACDSignal'], df['MACDHistogram'] = compute_macd(df['Close'])
    df['%K'], df['%D'] = util.compute_stochastic_oscillator(df['Close'], PARAMS['stochastic_oscillator'])

    ignore_first_ndays = max(PARAMS['volatility_tw'], PARAMS['days_price_changed'], PARAMS['bollinger_band'])
    X = df[['Volume',
            'Volatility',
            'Close',
            'RollingMean', 'UpperBand', 'LowerBand',
            'MACD', 'MACDSignal',
            '%K', '%D',
            ]].ix[ignore_first_ndays:]
    # print X
    y = (close.shift(-PARAMS['predict_time']) - close).apply(lambda x: -1 if x <= 0 else 1)
    y = y.ix[ignore_first_ndays:]
    X = X.ix[:len(y) - PARAMS['predict_time']]
    y = y.ix[:len(y) - PARAMS['predict_time']]
    return X, y, df.tail(LAST_N_DAYS)


def run_test():
    symbols = [
        'CTD', 'VNM', 'BMP', 'SKG', 'DRH', 'HSG', 'HPG', 'BVH', 'VIC', 'CAV'
    ]

    for symbol in symbols:
        print "-----------------------------"
        print "Predicting {}".format(symbol)
        X, y, last_n_records = prepare_train_data(symbol)
        predict_records = X.tail(5)

        x_train, x_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.15,
                                                            random_state=44)

        algorithms = [
            KNeighborsClassifier(n_neighbors=3),
            RandomForestClassifier(n_estimators=1000, n_jobs=-1),
            SVC(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(n_estimators=100),
            # LinearRegression(normalize=True),
        ]

        predicted_results = []
        for algo in algorithms:
            clf = algo
            clf.fit(x_train, y_train)
            predict_score = clf.score(x_test, y_test)
            predicted_result = clf.predict(predict_records.as_matrix())
            predicted_result = [x * predict_score for x in predicted_result]
            predicted_results.append(predicted_result)
        scores = np.matrix(predicted_results).sum(axis=0).tolist()[0]
        dates = last_n_records['Date'].as_matrix()
        for i in range(LAST_N_DAYS):
            print "{}: {:.2f}".format(dates[i], scores[i])


if __name__ == '__main__':
    run_test()
