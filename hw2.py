import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf
from bs4 import BeautifulSoup
from main import *


def get_tickets_from_url(URL):
    names = []
    response = requests.get(URL)
    print('response status = ', response.status_code)
    soup = BeautifulSoup(response.content, "html5lib")
    items = soup.findAll('a','apply-common-tooltip tickerNameBox-GrtoTeat tickerName-GrtoTeat')
    for item in items:
        names.append(item.text)
    print('number of stocks = ', len(names))

    return names

#короче берем какие то 50 активов каким то образом, в лог доходности, а потом сами крч пишите
#
num_assets=50 #количество активов
profitability= #считаем доходность
mean= #ожидаемая доходность
cov_matrix = profitability.cov()

#считаем риск для коэффициента Шарпа
def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def minimize_risk_with_short_sales():
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # это ограничение, сумма весов = 1
    bounds = tuple((None, None) for x in range(num_assets))  # это границы, разрешение коротких продаж
    initializer=num_assets * [1./num_assets,] #начальные веса
    optimal_sharpe_ss =minimize(portfolio_risk, initializer, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def minimize_risk_without_short_sales():
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))  # запрет коротких продаж
    initializer=num_assets * [1./num_assets,]
    result = minimize(portfolio_risk, initializer, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def get_data(URL,tickets_file,stocks_file):
    if not os.path.exists(tickets_file) or os.stat(tickets_file).st_size == 0:
        print('parcing tickets')
        tickets=get_tickets_from_url(URL)
        save_names_to_file(tickets,tickets_file)
        print('parcing tickets done')
    if not os.path.exists(stocks_file) or os.stat(stocks_file).st_size == 0:
        print('downloading stock data')
        tickets=get_names_from_file(tickets_file)
        download_stocks_to_excel(tickets,stocks_file)
        delete_null_columns(stocks_file)
        print('downloading stock data done')
    print('got data')
    print('stocks downloaded =',len(pd.read_excel(stocks_file).columns))



URL='https://ru.tradingview.com/symbols/NASDAQ-NDX/components/'
tickets_file='data2/tickets.txt'
stocks_file='data2/stocks.xlsx'

get_data(URL,tickets_file,stocks_file)

