import string

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
import numpy as np


def get_names_from_url(URL):
    names = []
    alphabet = list(string.ascii_uppercase)
    for letter in alphabet:
        new_URL = URL + letter + '.htm'
        response = requests.get(new_URL)
        print('response status = ', response.status_code)
        soup = BeautifulSoup(response.content, "html5lib")
        # print(page)
        items = soup.findAll('tr', 're')
        for item in items:
            names.append(item.td.text)
        items = soup.findAll('tr', 'ro')
        for item in items:
            names.append(item.td.text)
    print('number of stocks = ', len(names))
    return names


def save_names_to_file(names, file):
    with open(file, 'w'):
        pass

    with open(file, 'w') as fp:
        fp.write(' '.join(names))


def get_names_from_file(file):
    with open(file) as fp:
        names = fp.read().split(' ')
    return names


def download_stocks_to_excel(names, xlsx_file):
    data = pd.DataFrame(columns=names)
    for name in names:
        data[name] = yf.download(name, '2016-01-01', '2016-12-31')['Adj Close']

    data.to_excel(xlsx_file, index=False)
    return data.head()


def profitability(file_in, file_out):
    data = pd.read_excel(file_in)
    for i in range(len(data.columns)):
        data.iloc[:, i] = np.log1p(data.iloc[:, i].pct_change())
    data.to_excel(file_out, index=False)


if __name__ == '__main__':
    URL = 'https://www.eoddata.com/stocklist/NASDAQ/'
    names = get_names_from_url(URL)

    file = 'names.txt'
    save_names_to_file(names, file)
    names = get_names_from_file(file)

    input_file = 'input.xlsx'
    download_stocks_to_excel(names, input_file)

    names = pd.read_excel(input_file).head()
    save_names_to_file(names, file)

    pr_file = 'profitability.xlsx'
    profitability(input_file, pr_file)
    # in xlsx f5,choose all blank and delete columns = 1902 stocks instead of 4817
