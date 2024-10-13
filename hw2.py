import scipy.optimize as optimize
from pypfopt import plotting
from pypfopt.efficient_frontier import EfficientFrontier  # pip install pyportfolioopt
from scipy.optimize import minimize

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


def calculate_cov(file_in, file_out):
    if not os.path.exists(file_out) or os.stat(file_out).st_size == 0:
        df = pd.read_excel(file_in)
        cov_matrix = df.cov()
        cov_matrix.to_excel(file_out, index=False)


def create_bar_graph_weight(optimal_weights):
    plt.figure(figsize=(10, 6))
    tickets = range(0, 50)
    plt.bar(tickets, optimal_weights, color=['red'])
    plt.title('Сравнение весов акций')
    plt.ylabel('Вес акции')
    plt.show()


def create_bar_graph_risks(risk_no_short, risk_short):
    plt.figure(figsize=(10, 6))
    plt.bar(['С короткими', 'Без коротких'], [risk_short, risk_no_short], color=['blue', 'orange'])
    plt.title('Сравнение рисков портфелей')
    plt.ylabel('Стандартное отклонение (Риск)')
    plt.show()


# считаем риск для коэффициента Шарпа
def portfolio_risk(weights, cov_file):
    cov_matrix = pd.read_excel(cov_file)
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


def efficient_frontier(cov_file, mv_file):
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


def portfolio_optimization(mean_returns, cov_matrix, target_return):
    num_assets = len(mean_returns)
    weights = np.ones(num_assets) / num_assets

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'eq', 'fun': lambda weights: np.dot(weights, mean_returns) - target_return})

    bounds = tuple((0, 1) for asset in range(num_assets))

    result = minimize(portfolio_volatility, weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result


def find_10stocks(tickets_file10, mean_var50, stocks_file50, pr_file50, stocks_file10, pr_file10, mean_var10,
                  cov_matrix,
                  target_return=0.05):
    if not os.path.exists(tickets_file10) or os.stat(tickets_file10).st_size == 0:
        df = pd.read_excel(mean_var50)
        mean_returns = df['Мат ожидание'].values

        cov_matrix = pd.read_excel(cov_matrix).values

        optimal_portfolio = portfolio_optimization(mean_returns, cov_matrix, target_return)
        selected_stocks = df['Название акции'].iloc[optimal_portfolio['x'].argsort()[-10:]].values
        save_names_to_file(selected_stocks, tickets_file10)

    top_10 = get_names_from_file(tickets_file10)

    if not os.path.exists(stocks_file10) or os.stat(stocks_file10).st_size == 0:
        df2 = pd.read_excel(stocks_file50)
        columns = df2.columns[df2.columns.isin(top_10)]
        df2 = df2[columns]
        df2.to_excel(stocks_file10, index=False)

    if not os.path.exists(pr_file10) or os.stat(pr_file10).st_size == 0:
        df2 = pd.read_excel(pr_file50)
        columns = df2.columns[df2.columns.isin(top_10)]
        df2 = df2[columns]
        df2.to_excel(pr_file10, index=False)

    if not os.path.exists(mean_var10) or os.stat(mean_var10).st_size == 0:
        df2 = pd.read_excel(mean_var50)
        df2 = df2[df2['Название акции'].isin(top_10)]
        df2.to_excel(mean_var10, index=False)

    print(len(get_names_from_file(tickets_file10)))


def plot_efficient_frontier(returns, cov, color, label, ax, weight_bounds):
    ef = EfficientFrontier(returns, cov, weight_bounds=weight_bounds)
    plotting.plot_efficient_frontier(ef, ax=ax, ef_param_range=np.linspace(0.00, 0.006, 100), c=color)
    return ef


def plot_stocks(returns, cov, ax, color):
    # Предполагается, что returns содержит доходности акций
    ax.scatter(np.sqrt(np.diag(cov)), returns['Мат ожидание'], color=color)  # Точки на графике


def compare_efficient_frontiers(cov_file_50, mv_file_50, cov_file_10, mv_file_10):
    # Загружаем данные для полного набора акций (50 акций)
    returns_full = pd.read_excel(mv_file_50)
    cov_full = pd.read_excel(cov_file_50)

    # Загружаем данные для выбранного набора активов (10 акций)
    returns_selected = pd.read_excel(mv_file_10)
    cov_selected = pd.read_excel(cov_file_10)

    fig, ax = plt.subplots()

    # Эффективный фронт с короткими продажами (50 акций)
    ef_50 = plot_efficient_frontier(returns_full['Мат ожидание'], cov_full, color='blue',
                                    label='50 акций (Короткие продажи разрешены)', ax=ax, weight_bounds=(-1, 1))
    plot_stocks(returns_full, cov_full, ax, color='blue')  # Точки для 50 акций

    # Эффективный фронт с короткими продажами (10 акций)
    ef_10 = plot_efficient_frontier(returns_selected['Мат ожидание'], cov_selected, color='green',
                                    label='10 акций (Короткие продажи разрешены)', ax=ax, weight_bounds=(-1, 1))
    plot_stocks(returns_selected, cov_selected, ax, color='green')  # Точки для 10 акций

    plt.title('Эффективные фронты и акции (Короткие продажи разрешены)')
    plt.xlabel('Риск (стандартное отклонение)')
    plt.ylabel('Ожидаемая доходность')

    # Обновленная легенда
    lines = [plt.Line2D([0], [0], color='blue', lw=2),
             plt.Line2D([0], [0], color='green', lw=2)]
    plt.legend(lines, ['Эффективный фронт (50 акций)', 'Эффективный фронт (10 акций)'], title='Набор активов',
               loc='upper left')

    plt.show()

    fig, ax = plt.subplots()

    # Эффективный фронт без коротких продаж (50 акций)
    ef_50_no_short = plot_efficient_frontier(returns_full['Мат ожидание'], cov_full, color='blue',
                                             label='50 акций (Короткие продажи запрещены)', ax=ax, weight_bounds=(0, 1))
    plot_stocks(returns_full, cov_full, ax, color='blue')  # Точки для 50 акций

    # Эффективный фронт без коротких продаж (10 акций)
    ef_10_no_short = plot_efficient_frontier(returns_selected['Мат ожидание'], cov_selected, color='green',
                                             label='10 акций (Короткие продажи запрещены)', ax=ax, weight_bounds=(0, 1))
    plot_stocks(returns_selected, cov_selected, ax, color='green')  # Точки для 10 акций

    plt.title('Эффективные фронты и акции (Короткие продажи запрещены)')
    plt.xlabel('Риск (стандартное отклонение)')
    plt.ylabel('Ожидаемая доходность')

    # Обновленная легенда
    plt.legend(lines, ['Эффективный фронт (50 акций)', 'Эффективный фронт (10 акций)'], title='Набор активов',
               loc='upper left')

    plt.show()


if __name__ == '__main__':
    URL = 'https://ru.tradingview.com/symbols/NASDAQ-NDX/components/'
    tickets_file = 'data2/tickets.txt'
    stocks_file = 'data2/stocks.xlsx'
    pr_file2 = 'data2/profitability.xlsx'
    mean_var = 'data2/mean_var.xlsx'
    cov_file50 = 'data2/cov_file.xlsx'

    tickets_file50 = 'data2/tickets50.txt'
    stocks_file50 = 'data2/stocks50.xlsx'
    pr_file50 = 'data2/profitability50.xlsx'
    mean_var50 = 'data2/mean_var50.xlsx'
    pareto50 = 'data2/pareto50.xlsx'

    portfolio_min_risk_short_file = 'data2/portfolio_min_risk_short.xlsx'
    portfolio_min_risk_no_short_file = 'data2/portfolio_min_risk_no_short.xlsx'

    min_max_portfolio_return_short_file = 'data2/min_max_port_return_short.xlsx'
    ef_short_file50 = 'data2/ef_short.xlsx'
    ef_no_short_file50 = 'data2/ef_no_short.xlsx'

    download_data(URL, tickets_file, stocks_file)
    profitability(stocks_file, pr_file2)
    calculate_mean_var(pr_file2, mean_var)

    find_50stocks(stocks_file, pr_file2, mean_var, tickets_file50, stocks_file50, pr_file50, mean_var50)
    # find_pareto(mean_var50, pareto50)
    # create_mean_var_graphic(mean_var50, pareto50)
    num_assets = len(pd.read_excel(pr_file50).columns)
    calculate_cov(pr_file50, cov_file50)
    #
    # # портфель с минимальным риском с разрешением коротких продаж
    # minimize_risk_with_short_sales(cov_file50, portfolio_min_risk_short_file)
    # weights_min_risk_short = pd.read_excel(portfolio_min_risk_short_file)[0]
    # port_min_risk_return_short, port_min_risk_vol_short, sharpe_min_risk_short = portfolio(weights_min_risk_short,
    #                                                                                        pr_file50)
    # create_bar_graph_weight(weights_min_risk_short)
    #
    # # портфель с минимальным риском с запретом коротких продаж
    # minimize_risk_without_short_sales(cov_file50, portfolio_min_risk_no_short_file)
    # weights_min_risk_no_short = pd.read_excel(portfolio_min_risk_no_short_file)[0]
    # port_min_risk_return_no_short, port_min_risk_vol_no_short, sharpe_min_risk_no_short = portfolio(
    #     weights_min_risk_no_short, pr_file50)
    # create_bar_graph_weight(weights_min_risk_no_short)
    #
    # create_bar_graph_risks(port_min_risk_vol_no_short, port_min_risk_vol_short)
    # create_portfolio_graph(port_min_risk_vol_short, port_min_risk_return_short, port_min_risk_vol_no_short,
    #                        port_min_risk_return_no_short)
    #
    # # вычисляем эффективный фронт
    # efficient_frontier_short(cov_file50, mean_var50, ef_short_file50)
    # efficient_frontier_no_short(cov_file50, mean_var50, ef_no_short_file50)

    # efficient_frontier(cov_file50, mean_var50)

    tickets_file10 = 'data2/tickets10.txt'
    stocks_file10 = 'data2/stocks10.xlsx'
    pr_file10 = 'data2/profitability10.xlsx'
    mean_var10 = 'data2/mean_var10.xlsx'

    cov_file10 = 'data2/cov_file10.xlsx'

    ef_short_file10 = 'data2/ef_short.xlsx'
    ef_no_short_file10 = 'data2/ef_no_short.xlsx'

    find_10stocks(tickets_file10, mean_var50, stocks_file50, pr_file50, stocks_file10, pr_file10, mean_var10,
                  cov_file50)

    calculate_cov(pr_file10, cov_file10)
    # efficient_frontier_short(cov_file10, mean_var10, ef_short_file10)
    # efficient_frontier_no_short(cov_file10, mean_var10, ef_no_short_file10)
    #
    # efficient_frontier(cov_file10, mean_var10)
    compare_efficient_frontiers(cov_file50, mean_var50, cov_file10, mean_var10)
