import pandas as pd

from main import *


def get_tickets_from_url(URL):
    names = []
    response = requests.get(URL)
    print('response status = ', response.status_code)
    soup = BeautifulSoup(response.content, "html5lib")
    items = soup.findAll('a', 'apply-common-tooltip tickerNameBox-GrtoTeat tickerName-GrtoTeat')
    for item in items:
        names.append(item.text)
    print('number of stocks = ', len(names))

    return names


def get_data(URL, tickets_file, stocks_file):
    if not os.path.exists(tickets_file) or os.stat(tickets_file).st_size == 0:
        print('parcing tickets')
        tickets = get_tickets_from_url(URL)
        save_names_to_file(tickets, tickets_file)
        print('parcing tickets done')
    if not os.path.exists(stocks_file) or os.stat(stocks_file).st_size == 0:
        print('downloading stock data')
        tickets = get_names_from_file(tickets_file)
        download_stocks_to_excel(tickets, stocks_file)
        delete_null_columns(stocks_file)
        print('downloading stock data done')
        print('got data')
        print('stocks downloaded =', len(pd.read_excel(stocks_file).columns))
    print(len(get_names_from_file(tickets_file)))
    print(len(pd.read_excel(stocks_file).columns))

def calculate_cov(file_in, file_out):
    if not os.path.exists(cov_file) or os.stat(cov_file).st_size == 0:
        df = pd.read_excel(file_in)
        # result = []
        #
        # cov_num = np.cov(df['Мат ожидание'])
        # return cov_num
        cov_matrix = df.cov()
        cov_matrix.to_excel(file_out, index=False)


# считаем риск для коэффициента Шарпа
def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

'''
def minimize_risk_with_short_sales():
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # это ограничение, сумма весов = 1
    bounds = tuple((None, None) for x in range(num_assets))  # это границы, разрешение коротких продаж
    initializer = num_assets * [1. / num_assets, ]  # начальные веса
    result = minimize(portfolio_risk, initializer, args=(cov_matrix,),
                                 method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def minimize_risk_without_short_sales():
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))  # запрет коротких продаж
    initializer = num_assets * [1. / num_assets, ]
    result = minimize(portfolio_risk, initializer, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
'''

def find_50stocks(stocks_file,pr_file2, mean_var_file,tickets_file50, stocks_file50,pr_file52,mean_var50):
    if not os.path.exists(tickets_file50) or os.stat(tickets_file50).st_size == 0:
        df = pd.read_excel(mean_var_file)
        acceptable=df[(df['Мат ожидание']>0.0001) & (df['Дисперсия']<0.001)]
        acceptable['Соотношение']=acceptable['Мат ожидание']/acceptable['Дисперсия']
        acceptable=acceptable.sort_values(by='Соотношение',ascending=False)
        acceptable50=acceptable['Название акции'].head(50)
        save_names_to_file(acceptable50, tickets_file50)

    acceptable50=get_names_from_file(tickets_file50
                                     )
    if not os.path.exists(stocks_file50) or os.stat(stocks_file50).st_size == 0:
        df2=pd.read_excel(stocks_file)
        columns=df2.columns[df2.columns.isin(acceptable50)]
        df2=df2[columns]
        df2.to_excel(stocks_file50, index=False)

    if not os.path.exists(pr_file52) or os.stat(pr_file52).st_size == 0:
        df2=pd.read_excel(pr_file2)
        columns=df2.columns[df2.columns.isin(acceptable50)]
        df2=df2[columns]
        df2.to_excel(pr_file52, index=False)

    if not os.path.exists(mean_var50) or os.stat(mean_var50).st_size == 0:
        df2=pd.read_excel(mean_var_file)
        df2=df2[df2['Название акции'].isin(acceptable50)]
        df2.to_excel(mean_var50, index=False)

    print(len(get_names_from_file(tickets_file50)))

if __name__ == '__main__':
    URL = 'https://ru.tradingview.com/symbols/NASDAQ-NDX/components/'
    tickets_file = 'data2/tickets.txt'
    stocks_file = 'data2/stocks.xlsx'
    pr_file2 = 'data2/profitability.xlsx'
    mean_var = 'data2/mean_var.xlsx'
    cov_file = 'data2/cov_file.xlsx'

    tickets_file50='data2/tickets50.txt'
    stocks_file50='data2/stocks50.xlsx'
    pr_file52 = 'data2/profitability50.xlsx'
    mean_var50 = 'data2/mean_var50.xlsx'

    get_data(URL, tickets_file, stocks_file)

    if not os.path.exists(pr_file2) or os.stat(pr_file2).st_size == 0:
        profitability(stocks_file, pr_file2)

    if not os.path.exists(mean_var) or os.stat(mean_var).st_size == 0:
        calculate_mean_var(pr_file2, mean_var)

    find_50stocks(stocks_file,pr_file2, mean_var,tickets_file50,stocks_file50,pr_file52,mean_var50)

    num_assets = len(pd.read_excel(pr_file2).columns)

    calculate_cov(pr_file52, cov_file)
