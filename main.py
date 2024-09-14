import string

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


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


def get_data(input_file):
    URL = 'https://www.eoddata.com/stocklist/NASDAQ/'
    names = get_names_from_url(URL)

    file_names = 'names.txt'
    names = update_file_names(names, file_names)

    input_file = 'data/input.xlsx'
    download_stocks_to_excel(names, input_file)  # вот это очень долго работает, так что коммент этого делай
    # in xlsx f5,choose all blank and delete columns = 1902 stocks instead of 4817
    ### собрали все данные и ок ###

    save_names_to_file(pd.read_excel(input_file).head(), file_names)


def save_names_to_file(names, file):
    with open(file, 'w'):
        pass

    with open(file, 'w') as fp:
        fp.write(' '.join(names))


def get_names_from_file(file):
    with open(file) as fp:
        names = fp.read().split(' ')
    return names


def update_file_names(in_file, out_file):
    save_names_to_file(in_file, out_file)
    return get_names_from_file(out_file)


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
    input_file = 'data/input.xlsx'
    pr_file = 'data/profitability.xlsx'

    get_data(input_file)
    profitability(input_file, pr_file)
