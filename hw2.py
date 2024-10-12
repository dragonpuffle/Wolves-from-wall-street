import scipy.optimize as optimize
from pypfopt import plotting
from pypfopt.efficient_frontier import EfficientFrontier  # pip install pyportfolioopt

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


def download_data(URL, tickets_file, stocks_file):
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
        cov_matrix = df.cov()
        cov_matrix.to_excel(file_out, index=False)


# считаем риск для коэффициента Шарпа
def portfolio_risk(weights, cov_file):
    cov_matrix = pd.read_excel(cov_file)
    # Убедитесь, что веса суммируются до 1 (нормализация весов)
    weights = np.array(weights)
    if not np.isclose(np.sum(weights), 1):
        raise ValueError("Сумма весов портфеля должна быть равна 1.")

    # Расчёт риска портфеля (стандартное отклонение)
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def minimize_risk_with_short_sales(cov_file, portfolio_short_file):
    if not os.path.exists(portfolio_short_file) or os.stat(portfolio_short_file).st_size == 0:
        cov_matrix = pd.read_excel(cov_file)
        num_assets = cov_matrix.shape[0]
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # это ограничение, сумма весов = 1
        bounds = tuple((-1, 1) for x in range(num_assets))  # это границы, разрешение коротких продаж
        initializer = num_assets * [1. / num_assets, ]  # начальные веса
        result = optimize.minimize(portfolio_risk, initializer, args=(cov_file,),
                                   method='SLSQP', bounds=bounds, constraints=constraints)
        res = pd.DataFrame(result.x)
        res.to_excel(portfolio_short_file)


def minimize_risk_without_short_sales(cov_file, portfolio_no_short_file):
    if not os.path.exists(portfolio_no_short_file) or os.stat(portfolio_no_short_file).st_size == 0:
        cov_matrix = pd.read_excel(cov_file)
        num_assets = cov_matrix.shape[0]
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))  # запрет коротких продаж
        initializer = num_assets * [1. / num_assets, ]
        result = optimize.minimize(portfolio_risk, initializer, args=(cov_file,),
                                   method='SLSQP', bounds=bounds, constraints=constraints)
        res = pd.DataFrame(result.x)
        res.to_excel(portfolio_no_short_file)


def portfolio(weights, returns_file):
    weights = np.array(weights)
    returns = pd.read_excel(returns_file)
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = port_return / port_vol
    return port_return, port_vol, sharpe


def portfolio_return(weights, returns_file):
    weights = np.array(weights)
    returns = pd.read_excel(returns_file)
    port_return = np.sum(returns.mean() * weights) * 252
    return port_return


def maximize_portfolio_return(weights, returns_file):
    return -portfolio_return(weights, returns_file)


def find_min_max_portfolio_return_short(cov_file, min_max_portfolio_return_short_file):
    if not os.path.exists(min_max_portfolio_return_short_file) or os.stat(
            min_max_portfolio_return_short_file).st_size == 0:
        cov_matrix = pd.read_excel(cov_file)
        num_assets = cov_matrix.shape[0]
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((-1, 1) for asset in range(num_assets))
        initializer = num_assets * [1. / num_assets, ]
        result1 = optimize.minimize(portfolio_return, initializer, args=(cov_file,),
                                    method='SLSQP', bounds=bounds, constraints=constraints)
        result2 = optimize.minimize(maximize_portfolio_return, initializer, args=(cov_file,),
                                    method='SLSQP', bounds=bounds, constraints=constraints)
        res = pd.DataFrame()
        res['min'] = result1.x
        res['max'] = result2.x

        res.to_excel(min_max_portfolio_return_short_file)


def efficient_frontier(cov_file, mv_file, pic):
    returns = pd.read_excel(mv_file)
    cov = pd.read_excel(cov_file)
    fig, ax = plt.subplots()
    ef = EfficientFrontier(returns['Мат ожидание'], cov, weight_bounds=(-1, 1))
    plotting.plot_efficient_frontier(ef, ax=ax, ef_param_range=np.linspace(0.00, 0.006, 100), c='blue', )
    plt.legend(['Короткие продажи\nразрешены', 'short assets'])
    plt.show()
    fig, ax = plt.subplots()
    ef = EfficientFrontier(returns['Мат ожидание'], cov, weight_bounds=(0, 1))
    plotting.plot_efficient_frontier(ef, ax=ax, c='green')
    plt.legend(['Короткие продажи запрещены', 'no short assets'])
    plt.show()


def efficient_frontier_short(cov_file, mv_file, ef_short_file):
    if not os.path.exists(ef_short_file) or os.stat(ef_short_file).st_size == 0:
        returns = pd.read_excel(mv_file)
        cov = pd.read_excel(cov_file)
        ef = EfficientFrontier(returns['Мат ожидание'], cov, weight_bounds=(-1, 1))
        minvol = ef.min_volatility()
        weights = ef.clean_weights()
        res = pd.DataFrame()
        res['ticket'] = returns['Название акции']
        res['weights'] = weights
        res.to_excel(ef_short_file)


def efficient_frontier_no_short(cov_file, mv_file, ef_no_short_file):
    if not os.path.exists(ef_no_short_file) or os.stat(ef_no_short_file).st_size == 0:
        returns = pd.read_excel(mv_file)
        cov = pd.read_excel(cov_file)
        ef = EfficientFrontier(returns['Мат ожидание'], cov, weight_bounds=(0, 1))
        minvol = ef.min_volatility()
        weights = ef.clean_weights()
        res = pd.DataFrame()
        res['ticket'] = returns['Название акции']
        res['weights'] = weights
        res.to_excel(ef_no_short_file)


def find_50stocks(stocks_file, pr_file2, mean_var_file, tickets_file50, stocks_file50, pr_file52, mean_var50,
                  risk_free_rate=0.01):
    if not os.path.exists(tickets_file50) or os.stat(tickets_file50).st_size == 0:
        df = pd.read_excel(mean_var_file)
        # Убираем акции с отрицательным мат ожиданием и слишком высокой дисперсией
        acceptable = df[(df['Мат ожидание'] > 0.0001) & (df['Дисперсия'] < 0.001)]
        # Рассчитываем коэффициент Шарпа (Sharpe Ratio = (Мат ожидание - Безрисковая ставка) / Стандартное отклонение)
        acceptable['Sharpe Ratio'] = (acceptable['Мат ожидание'] - risk_free_rate) / acceptable['Дисперсия'] ** 0.5
        acceptable = acceptable.sort_values(by='Sharpe Ratio', ascending=False)
        acceptable50 = acceptable['Название акции'].head(50)
        save_names_to_file(acceptable50, tickets_file50)

    acceptable50 = get_names_from_file(tickets_file50
                                       )
    if not os.path.exists(stocks_file50) or os.stat(stocks_file50).st_size == 0:
        df2 = pd.read_excel(stocks_file)
        columns = df2.columns[df2.columns.isin(acceptable50)]
        df2 = df2[columns]
        df2.to_excel(stocks_file50, index=False)

    if not os.path.exists(pr_file52) or os.stat(pr_file52).st_size == 0:
        df2 = pd.read_excel(pr_file2)
        columns = df2.columns[df2.columns.isin(acceptable50)]
        df2 = df2[columns]
        df2.to_excel(pr_file52, index=False)

    if not os.path.exists(mean_var50) or os.stat(mean_var50).st_size == 0:
        df2 = pd.read_excel(mean_var_file)
        df2 = df2[df2['Название акции'].isin(acceptable50)]
        df2.to_excel(mean_var50, index=False)

    print(len(get_names_from_file(tickets_file50)))


def create_mean_var_graphic(mv_file, pareto_file):
    df_mv = pd.read_excel(mv_file)
    pareto = pd.read_excel(pareto_file)
    fig, ax = plt.subplots()

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
    ax.legend()
    plt.show()


def create_bar_graph_risks(risk_no_short, risk_short):
    plt.figure(figsize=(10, 6))
    plt.bar(['С короткими', 'Без коротких'], [risk_short, risk_no_short], color=['blue', 'orange'])
    plt.title('Сравнение рисков портфелей')
    plt.ylabel('Стандартное отклонение (Риск)')
    plt.show()


def create_bar_graph_weight(optimal_weights):
    plt.figure(figsize=(10, 6))
    tickets = range(0, 50)
    plt.bar(tickets, optimal_weights, color=['red'])
    plt.title('Сравнение весов акций')
    plt.ylabel('Вес акции')
    plt.show()


def create_portfolio_graph(risk_short, mean_short, risk_no_short, mean_no_short):
    fig, ax = plt.subplots()
    ax.scatter(
        risk_short,
        mean_short,
        marker='o',
        color='blue',
        s=80,  # Размер маркера
        label='short'
    )

    ax.scatter(
        risk_no_short,
        mean_no_short,
        marker='o',
        color='green',
        s=80,
        label='no_short'
    )
    plt.title('Сравнение портфелей')
    plt.legend()
    plt.ylabel('mean')
    plt.xlabel('var')
    plt.show()


def create_portfolios_graph(eff_front_file):
    df = pd.read_excel(eff_front_file)
    port_vols = df['var']
    port_returns = df['mean']
    plt.figure(figsize=(18, 10))
    plt.scatter(port_vols,
                port_returns,
                c=(port_returns / port_vols),
                marker='o')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label='Sharpe ratio (not adjusted for short rate)')


if __name__ == '__main__':
    URL = 'https://ru.tradingview.com/symbols/NASDAQ-NDX/components/'
    tickets_file = 'data2/tickets.txt'
    stocks_file = 'data2/stocks.xlsx'
    pr_file2 = 'data2/profitability.xlsx'
    mean_var = 'data2/mean_var.xlsx'
    cov_file = 'data2/cov_file.xlsx'

    tickets_file50 = 'data2/tickets50.txt'
    stocks_file50 = 'data2/stocks50.xlsx'
    pr_file52 = 'data2/profitability50.xlsx'
    mean_var50 = 'data2/mean_var50.xlsx'
    pareto50 = 'data2/pareto50.xlsx'

    portfolio_min_risk_short_file = 'data2/portfolio_min_risk_short.xlsx'
    portfolio_min_risk_no_short_file = 'data2/portfolio_min_risk_no_short.xlsx'

    min_max_portfolio_return_short_file='data2/min_max_port_return_short.xlsx'
    ef_short_file='data2/ef_short.xlsx'
    ef_no_short_file = 'data2/ef_no_short.xlsx'
    pic='data2/Efficient_frontiers.png'

    download_data(URL, tickets_file, stocks_file)
    profitability(stocks_file, pr_file2)
    calculate_mean_var(pr_file2, mean_var)

    find_50stocks(stocks_file, pr_file2, mean_var, tickets_file50, stocks_file50, pr_file52, mean_var50)
    find_pareto(mean_var50, pareto50)
    create_mean_var_graphic(mean_var50, pareto50)
    num_assets = len(pd.read_excel(pr_file52).columns)
    calculate_cov(pr_file52, cov_file)
    '''
    minimize_risk_with_short_sales(cov_file, portfolio_min_risk_short_file)
    weights_min_risk_short = pd.read_excel(portfolio_min_risk_short_file)[0]
    port_min_risk_return_short, port_min_risk_vol_short, sharpe_min_risk_short = portfolio(weights_min_risk_short, pr_file52)
    print(port_min_risk_return_short, port_min_risk_vol_short, sharpe_min_risk_short)
    create_bar_graph_weight(weights_min_risk_short)
    
    minimize_risk_without_short_sales(cov_file, portfolio_min_risk_no_short_file)
    weights_min_risk_no_short = pd.read_excel(portfolio_min_risk_no_short_file)[0]
    port_min_risk_return_no_short, port_min_risk_vol_no_short, sharpe_min_risk_no_short = portfolio(weights_min_risk_no_short, pr_file52)
    print(port_min_risk_return_no_short, port_min_risk_vol_no_short, sharpe_min_risk_no_short)
    create_bar_graph_weight(weights_min_risk_no_short)
    
    create_bar_graph_risks(port_min_risk_vol_no_short, port_min_risk_vol_short)
    create_portfolio_graph(port_min_risk_vol_short, port_min_risk_return_short, port_min_risk_vol_no_short, port_min_risk_return_no_short)
    '''
    #efficient_frontier_short(cov_file,mean_var50,ef_short_file)
    #efficient_frontier_no_short(cov_file, mean_var50, ef_no_short_file)

    efficient_frontier(cov_file,mean_var50,pic)