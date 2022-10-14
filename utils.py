from statsmodels.tsa.stattools import adfuller

def is_stationary(data):
    result = adfuller(data)
    pvalue = result[1]

    if pvalue < 0.05:
        return True
    else:
        return False