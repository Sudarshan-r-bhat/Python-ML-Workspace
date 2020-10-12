import pandas as pd
from pandas import read_csv
import os
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib import style
import re


style.use('dark_background')
# from tqdm import tqdm # create an obj to pass params: unit, unit_scale, desc

path = 'D:/ML_datasets/ML_share_dataset/intraQuarter/intraQuarter'


def key_stats(gather="Total Debt/Equity (mrq)"):
    statspath = path + '/_KeyStats'
    stock_list = [x[0] for x in os.walk(statspath)]  # we add all directories present in the _Key_Stats folder. it returns a tuple of 3 elements. folder, subfolders, files.
    print(stock_list)

    df = pd.DataFrame(columns=['date',
                               'unix',
                               'ticker',
                               'DE ratio',
                               'price',
                               'stock_p_change',
                               'sp500',
                               'sp500_p_change',
                               'difference',
                               'status'
                               ])  # the values in the dataframe are stored as dictionary values. keys represent the labels.


    sp500_df = read_csv("D:\\ML_datasets\\ML_share_dataset\\sandp500\\YAHOO-INDEX_GSPC.csv")
    ticker_list = []
    for each_dir in stock_list[1:]:
        # print('each_dir', each_dir)
        each_file = os.listdir(each_dir)  # listdir() will list whatever is in the current directory.
        # print('each_file', each_file)
        ticker = each_dir.split('\\')[1]  # name of each folder containing html files.
        ticker_list.append(ticker)

        starting_stock_value = False
        starting_sp500_value = False
        if len(each_file) > 0:
            for file in each_file:
                date_stamp = datetime.datetime.strptime(file, '%Y%m%d%H%M%S.html')
                unix_time = time.mktime(date_stamp.timetuple())
                full_file_path = each_dir + '/' + file
                source = open(full_file_path, 'r').read()

                try:
                    try:
                        data = float(source.split(gather)[1].split('<td class="yfnc_tabledata1">')[1].split('</td>')[0])  # this is how i parse the required data.
                        print(data)
                    except Exception as e:
                        print(str(e))
                        time.sleep(3)
                        pass
                    try:
                        sp500_date = datetime.datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
                        row = sp500_df.loc[(sp500_df['Date'] == sp500_date)]  # the error is csv not taken into dataframe.
                        sp500_value = float(row['Adjusted Close'])
                        print('the sp500 value is: ', sp500_value, 'the row value is: ', row)
                    except Exception as e:
                        sp500_date = datetime.datetime.fromtimestamp(unix_time - 259200).strftime('%Y-%m-%d')
                        row = sp500_df[(sp500_df.index == sp500_date)]
                        sp500_value = float(row['Adjusted Close'])
                        print('we are in the except block- ', sp500_value)

                    stock_price = float(source.split('</small><big><b>')[1].split('</b></big>')[0])

                    if not starting_stock_value:
                        starting_stock_value = stock_price
                    if not starting_sp500_value:
                        starting_sp500_value = sp500_value

                    stock_p_change = ((stock_price - starting_stock_value) / starting_stock_value) * 100
                    sp500_p_change = ((sp500_value - starting_sp500_value) / starting_sp500_value) * 100

                    difference = stock_p_change - sp500_p_change
                    status = ''
                    if difference > 0:
                        status = 'outperform'
                    else:
                        status = 'underperform'

                    df = df.append({'Date': date_stamp,
                                    'unix': unix_time,
                                    'ticker': ticker,
                                     'DE ratio': data,
                                    'price': price,
                                    'stock_p_change': stock_p_change,
                                    'sp500': sp500,
                                    'sp500_p_change': sp500_p_change,
                                    'difference': stock_p_change - sp500_p_change,
                                    'status': status},
                                   ignore_index=True)


                except Exception as e:
                    print('exception occured', e.with_traceback(None))
                    pass

    for each_ticker in ticker_list:
        try:
            plot_df = df.loc[:, [2 if (df.loc[:, 2] == each_ticker)]]# to access rows and cols in a pd.DataFrame df.loc[: , :] this means all rows and all cols, df.loc[:, [3]] means from all rows select 3rd column.
            plot_df = plot_df.set_index(['date'])

            if plot_df['status'][-1] == "underperform":
                color = 'r'
            else:
                color = 'g'

            plot_df['difference'].plot(label=each_ticker, color=color)
            plt.legend()
        except:
            pass

        plt.show()


    #
    # save = gather.replace(' ', '').replace(')', '').replace('(', '').replace('/', '') + '.csv'
    # print(save)
    # df.to_csv(save)
    #


key_stats()
