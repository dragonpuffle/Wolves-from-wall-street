import os
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

    download_stocks_to_excel(names, input_file)  # вот это очень долго работает, так что коммент этого делай
    # in xlsx f5,choose all blank and delete columns = 1902 stocks instead of 4817
    ### собрали все данные и ок ###
    delete_null_columns(input_file)
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


def profitability(file_in, file_out):
    data = pd.read_excel(file_in)
    for i in range(len(data.columns)):
        data.iloc[:, i] = np.log1p(data.iloc[:, i].pct_change())
    data.to_excel(file_out, index=False)


def delete_null_columns(file):
    df = pd.read_excel(file)
    df_cleaned = df.dropna(axis=1, how='any')
    df_cleaned.to_excel(file, index=False)


def calculate_mean_var(file_in, file_out):
    df = pd.read_excel(file_in)
    result = []

    for column in df.columns[1:]:
        stock_name = df[column].name
        mean = np.mean(df[column])  # Математическое ожидание
        variance = np.var(df[column])  # Дисперсия

        result.append({
            'Название акции': stock_name,
            'Мат ожидание': mean,
            'Дисперсия': variance})

    result_df = pd.DataFrame(result)

    result_df.to_excel(file_out, index=False)

def find_pareto(xlsx_file_in,txt_file_out):
    data = pd.read_excel(xlsx_file_in)
    data=data.sort_values(by=['Мат ожидание','Дисперсия'],ascending=False).reset_index(drop = True)
    pareto_stocks=[]
    previous_var=100
    for index, stock in data.iterrows():
        if stock['Дисперсия']<=previous_var:
            pareto_stocks.append(stock['Название акции'])
            previous_var=stock['Дисперсия']

    save_names_to_file(pareto_stocks,txt_file_out)

def calculate_historical_var(xlsx_file_in,txt_file_in,xlsx_file_out):
    data=pd.read_excel(xlsx_file_in)
    pareto=get_names_from_file(txt_file_in)
    vars = []
    for stock in pareto:
        profit=data[stock]
        profit=profit.sort_values().reset_index(drop = True)
        vars.append({'Stocks':stock,'Var':profit[12]})#95% от года= 12 дней

    vars_pd=pd.DataFrame(vars)
    vars_pd=vars_pd.sort_values('Var',ascending=False).reset_index(drop = True)#самое лучшее по var на 1 месте(если отриц, то теряем, если полож, то получим(почти невозможно))
    vars_pd.to_excel(xlsx_file_out,index=False)



if __name__ == '__main__':
    input_file = 'data/input.xlsx'
    pr_file = 'data/profitability.xlsx'
    mv_file = 'data/stock_results.xlsx'
    pareto_file='data/pareto_stocks.txt'
    vars_file='data/vars.xlsx'


    if not os.path.exists(input_file) or os.stat(input_file).st_size == 0:
        get_data(input_file)  # комментишь это и данные не собираются, хотя лучше просто сделать проверку

    if not os.path.exists(pr_file) or os.stat(pr_file).st_size == 0:
        profitability(input_file, pr_file)

    if not os.path.exists(mv_file) or os.stat(mv_file).st_size == 0:
        calculate_mean_var(pr_file, mv_file)

    if not os.path.exists(pareto_file) or os.stat(pareto_file).st_size == 0:
        find_pareto(mv_file, pareto_file)

    if not os.path.exists(vars_file) or os.stat(vars_file).st_size == 0:
        calculate_historical_var(pr_file,pareto_file,vars_file)

    print(len(get_names_from_file('names.txt')))