from pandas import read_csv
from pandas import to_datetime
from pandas import DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv'
data = read_csv(url, header=0, parse_dates=True, index_col=0, squeeze=True)
print(data.head())

_ = data.plot()
pyplot.show()

_ = autocorrelation_plot(data)
pyplot.show()

model = ARIMA(data, order=(2, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = DataFrame(model_fit.resid)
_ = residuals.plot()
pyplot.show()
_ = residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

X = data.values
size = int(len(X) * (2/3))
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

_ = pyplot.plot(test)
_ = pyplot.plot(predictions, color='red')
pyplot.show()