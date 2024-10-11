import cvxpy as cp

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


def portfolio_with_minimal_risk_short(cov_file):
    cov = pd.read_excel(cov_file)
    n = len(cov.columns)

    weights = cp.Variable(n)
    portfolio_variance = cp.quad_form(weights, cov)
    constraints = [cp.sum(weights) == 1]
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()
    optimal_weights_short = weights.value
    return optimal_weights_short, np.sqrt(problem.value), np.mean(problem.value)


def portfolio_with_minimal_risk_no_short(cov_file):
    cov = pd.read_excel(cov_file)
    n = len(cov.columns)

    weights = cp.Variable(n)
    portfolio_variance = cp.quad_form(weights, cov)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()
    optimal_weights_no_short = weights.value
    return optimal_weights_no_short, np.sqrt(problem.value), np.mean(problem.value)


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


def create_portfel_graph(risk_short, mean_short, risk_no_short, mean_no_short):
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

    download_data(URL, tickets_file, stocks_file)
    profitability(stocks_file, pr_file2)
    calculate_mean_var(pr_file2, mean_var)

    find_50stocks(stocks_file, pr_file2, mean_var, tickets_file50, stocks_file50, pr_file52, mean_var50)

    find_pareto(mean_var50, pareto50)
    create_mean_var_graphic(mean_var50, pareto50)

    num_assets = len(pd.read_excel(pr_file2).columns)

    calculate_cov(pr_file52, cov_file)

    optimal_weights_short, risk_short, mean_short = portfolio_with_minimal_risk_short(cov_file)
    create_bar_graph_weight(optimal_weights_short)
    optimal_weights_no_short, risk_no_short, mean_no_short = portfolio_with_minimal_risk_no_short(cov_file)
    create_bar_graph_weight(optimal_weights_no_short)
    create_bar_graph_risks(risk_no_short, risk_short)
    create_portfel_graph(risk_short, mean_short, risk_no_short, mean_no_short)
