import os
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf
from bs4 import BeautifulSoup
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from scipy.stats import shapiro, normaltest


def get_names_from_url(URL):
    names = []
    alphabet = list(string.ascii_uppercase)
    for letter in alphabet:
        new_URL = URL + letter + '.htm'
        response = requests.get(new_URL)
        print('response status = ', response.status_code)
        soup = BeautifulSoup(response.content, "html5lib")
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


def delete_null_columns(file):
    df = pd.read_excel(file)
    df_cleaned = df.dropna(axis=1, how='any')
    df_cleaned.to_excel(file, index=False)


def profitability(file_in, file_out):
    data = pd.read_excel(file_in)
    for i in range(len(data.columns)):
        data.iloc[:, i] = np.log1p(data.iloc[:, i].pct_change())
    data.to_excel(file_out, index=False)


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


def find_pareto(xlsx_file_in, xlx_file_out):
    data = pd.read_excel(xlsx_file_in)
    data = data.sort_values(by=['Мат ожидание', 'Дисперсия'], ascending=False).reset_index(drop=True)
    pareto_stocks = []
    previous_var = 100
    for index, stock in data.iterrows():
        if stock['Дисперсия'] <= previous_var and stock['Мат ожидание'] > 0:
            pareto_stocks.append({
                'Название акции': stock['Название акции'],
                'Мат ожидание': stock['Мат ожидание'],
                'Дисперсия': stock['Дисперсия']})
            previous_var = stock['Дисперсия']

    # save_names_to_file(pareto_stocks, txt_file_out)
    pd.DataFrame(pareto_stocks).to_excel(xlx_file_out, index=False)


def calculate_historical_var(xlsx_file_in, pareto_file_in, xlsx_file_out):
    data = pd.read_excel(xlsx_file_in)
    pareto = pd.read_excel(pareto_file_in)['Название акции']
    vars = []
    for stock in pareto:
        profit = data[stock]
        profit = profit.sort_values().reset_index(drop=True)
        vars.append({'Stocks': stock, 'Var': profit.iloc[12]})  # 95% от года= 12 дней

    vars_pd = pd.DataFrame(vars)
    vars_pd = vars_pd.sort_values('Var', ascending=False).reset_index(
        drop=True)  # самое лучшее по var на 1 месте(если отриц, то теряем, если полож, то получим(почти невозможно))
    vars_pd.to_excel(xlsx_file_out, index=False)


def calculate_cvar(xlsx_file_in, pareto_file_in, xlsx_file_out):
    data = pd.read_excel(xlsx_file_in)
    pareto = pd.read_excel(pareto_file_in)['Название акции']
    cvars = []
    for stock in pareto:
        profit = data[stock]
        profit = profit.sort_values().reset_index(drop=True)
        cvars.append({'Stocks': stock, 'Cvar': profit.iloc[0:12].mean()})

    cvars_pd = pd.DataFrame(cvars)
    cvars_pd = cvars_pd.sort_values('Cvar', ascending=False).reset_index(
        drop=True)  # лучшее по cvar на 1 месте, если отриц, то теряем, если положит, то получим (почти невозможно)
    cvars_pd.to_excel(xlsx_file_out, index=False)


def view_price(file):
    df = pd.read_excel(file, usecols=['GMGI'])
    sns.relplot(
        data=df,
        kind='line'
        # color = '#40f4ef',
    )
    plt.show()


def create_schedule(mv_file, pareto_file):
    df_mv = pd.read_excel(mv_file)
    pareto = pd.read_excel(pareto_file)

    point_x1_var = 2.84306137131399E-08
    point_y1_var = 0.0000163551398081471

    point_x2_var = 0.0753697111058486
    point_y2_var = 0.0152849221231872

    point_x3_var = 0
    point_y3_var = 0
    # Создаем фигуру и оси
    fig, ax = plt.subplots()

    # Строим первый график
    sns.scatterplot(
        data=df_mv,
        y='Мат ожидание',
        x='Дисперсия',
        ax=ax,
        color='#40f4ef',
        label='MV Data',
    )

    # Строим второй график
    sns.scatterplot(
        data=pareto,
        y='Мат ожидание',
        x='Дисперсия',
        ax=ax,
        s=100,
        marker='x',
        color='red',
        label='Pareto Data'
    )

    ax.scatter(
        point_x1_var,
        point_y1_var,
        marker='o',
        color='blue',
        s=80,  # Размер маркера
        label='SHV (cvar)'
    )

    ax.scatter(
        point_x2_var,
        point_y2_var,
        marker='o',
        color='green',
        s=80,
        label='CELZ (var)'
    )

    # ax.scatter(
    #     point_x3_var,
    #     point_y3_var,
    #     marker='o',  # Задаем новый маркер
    #     color='blue',  # Цвет для новой точки
    #     s=80,  # Размер маркера
    #     label='QUBT (cvar)'
    # )

    # Добавляем легенду
    ax.legend()
    plt.show()


def count_inversions(data):
    inversions = 0
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            if data[i] > data[j]:
                inversions += 1
    return inversions


def get_left_right_borders(data):
    n = len(data) - 1
    t = 0.062707
    t_n = t * (n ** (3 / 2)) / 6
    left = n * (n - 1) / 4 - t_n
    right = n * (n - 1) / 4 + t_n
    return left, right


def is_white_noise(points, file):
    df = pd.read_excel(file)
    for point in points:
        data = df[point]
        inversions = count_inversions(data)
        left, right = get_left_right_borders(data)
        if inversions > left and inversions < right:
            print(point, ' является белым шумом')
        else:
            print(point, ' не является белым шумом')
    print('------------------------------------------------')


def is_normal(points, file):
    df = pd.read_excel(file)
    for point in points:
        k2, p = stats.normaltest(df[point][1:])
        if p < 0.05:
            print(point, ' следует нормальному распределению')
        else:
            print(point, ' не следует нормальному распределению')
        print(point, shapiro(df[point][1:]), normaltest(df[point][1:]))
    print('------------------------------------------------')


def kdeplt(points, file):
    df = pd.read_excel(file)
    for point in points:
        sns.kdeplot(data=df[point][1:], common_norm=False)
    plt.show()


def histplt(points, file):
    df = pd.read_excel(file)
    for point in points:
        sns.histplot(data=df[point][1:], bins=len(df[point][1:]), stat="density", element="step", fill=False,
                     cumulative=True, common_norm=False)
    plt.show()


def count_volatility(points, file):
    df = pd.read_excel(file)
    for point in points:
        volatility = df[point].std()
        print(f'Стандартное отклонение доходностей: {volatility}')
    print('------------------------------------------------')


def find_anomalies(points, file):
    df = pd.read_excel(file)
    for point in points:
        q1 = df[point].quantile(0.25)
        q3 = df[point].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        anomalies = (df[point] < lower) | (df[point] > upper)
        print(f'Обнаруженные аномалии:\n{anomalies}')
    print('------------------------------------------------')


if __name__ == '__main__':
    input_file = 'data/input.xlsx'
    pr_file = 'data/profitability.xlsx'
    mv_file = 'data/stock_results.xlsx'
    pareto_file = 'data/pareto_stocks.xlsx'
    vars_file = 'data/vars.xlsx'
    cvars_file = 'data/cvars.xlsx'

    important_dots = ['GMGI', 'SHV', 'CELZ', 'BCDA']
    # important_dots = ['AAL',	'AAOI',	'AAPL',	'AAXJ',	'ABEO',	'ABTS',	'ABVC',	'ACAD',	'ACHC',	'ACHV',
    #                  'ACLS',	'ACNT',	'ACWI',	'ADAP',	'ADD',	'ADP',	'ADSK',	'AEHR',	'AEIS',	'AEP',
    #                 'AFMD',	'AGEN', 'AGIO',	'AGNC',	'AGYS',	'AIA',	'AIFF',	'AIRR',	'AIRT']
    different_dots = ['AAPL', 'TSLA', 'AZN', 'TMUS', 'LIN']

    if not os.path.exists(input_file) or os.stat(input_file).st_size == 0:
        get_data(input_file)  # комментишь это и данные не собираются, хотя лучше просто сделать проверку

    if not os.path.exists(pr_file) or os.stat(pr_file).st_size == 0:
        profitability(input_file, pr_file)

    if not os.path.exists(mv_file) or os.stat(mv_file).st_size == 0:
        calculate_mean_var(pr_file, mv_file)

    if not os.path.exists(pareto_file) or os.stat(pareto_file).st_size == 0:
        find_pareto(mv_file, pareto_file)

    if not os.path.exists(vars_file) or os.stat(vars_file).st_size == 0:
        calculate_historical_var(pr_file, pareto_file, vars_file)

    if not os.path.exists(cvars_file) or os.stat(cvars_file).st_size == 0:
        calculate_cvar(pr_file, pareto_file, cvars_file)

    # interesting_point(important_dots, pr_file)
    is_white_noise(important_dots, pr_file)
    is_normal(different_dots, pr_file)
    kdeplt(different_dots, pr_file)  # ПОЧЕМУ ТО ОН ЭТИ ДВЕ РАЗНЫЕ ФУНКЦИИ СТРОИТ НА ОДНОМ ХОЛСТЕ, ПОЧЕМУ
    histplt(different_dots, pr_file)  #
    count_volatility(different_dots, pr_file)
    find_anomalies(different_dots, pr_file)

    create_schedule(mv_file, pareto_file)
