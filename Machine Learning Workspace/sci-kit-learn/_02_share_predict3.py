import pandas as pd
import os
from Quandl import Quandl
import time

auth_token = "yourauthhere"
"""data = Quadl.get("WIKI/KO", trim_start="2000-12-12", trim_end="2014-12-30", authtoken=auth_tok)
print(data)"""

path = "D:\\ML_datasets\\ML_share_dataset\\intraQuarter\\intraQuarter\\"
df = pd.DataFrame()


def stock_prices():
    statspath = path + '_KeyStats'
    stock_list = [x[0] for x in os.walk(path)]

    for each_dir in stock_list[1:]:
        try:

            ticker = each_dir.split("\\[a-z]*\s*[a-z]*$")[1:]  # this will get only the names of the stocks from the dir.
            data = Quandl.get('WIKI/' + ticker.upper(), trim_start="2000-12-12", trim_end="2014-12-30", authtoken=auth_token)
            data[ticker.upper()] = data['Adjusted Close']
            df = pd.concat(df, [data[ticker.upper()]], axis=1)
        except Exception as e:
            print(str(e))
            time.sleep(10)
            try:                                                    # we try to fetch the data twice from quandl.com
                ticker = each_dir.split("\\[a-z]*\s*[a-z]*$")[1:]
                data = Quandl.get('WIKI/' + ticker.upper(), trim_start="2000-12-12", trim_end="2014-12-30", authtoken=auth_token)
                data[ticker.upper()] = data['Adjusted Close']
                df = pd.concat(df, [data[ticker.upper()]], axis=1)
            except Exception as e:
                print(str(e))
                pass
    df.to_csv(path + "stock_prices.csv")


stock_prices()

