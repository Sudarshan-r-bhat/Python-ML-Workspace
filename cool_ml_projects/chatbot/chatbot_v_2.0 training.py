import sqlite3
import pandas as pd
timeframes = ['2015-08']

for timeframe in timeframes:
    connection = sqlite3.connect('{}.db'.format(timeframe))
    c = connection.cursor()
    limit = 5000
    last_unix = 0
    cur_length = limit
    counter = 0
    test_done = False
    while cur_length == limit:
        df = pd.read_sql("SELECT * FROM parent_reply unix > {} AND parent NOT NULL AND score > 0 ORDER BY "
                         "unix ASC LIMIT {}".format(last_unix, limit), connection)
        last_unix = df.tail(1)['unix'].values[0]
        cur_length = len(df)
        if not test_done:
            with open('test.from', 'a', encoding='utf-8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')
            with open('test.to', 'a', encoding='utf-8') as f:
                for content in df['comment'].values:
                    f.write(content + '\n')
            test_done = True
        else:
            with open('train.from', 'a', encoding='utf-8') as f:
                for content in df['parent'].values:
                    f.write(content + '\n')
            with open('train.to', 'a', encoding='utf-8') as f:
                for content in df['comment'].values:
                    f.write(content + '\n')

        counter += 1
        if counter % 20 == 0:
            print(counter * limit, 'rows completed so far')

if '__name__' == '__main__':
    output_file_location = 'output_dev'
    tst_file_location = 'tst_2018.from'
    with open(output_file_location, 'r') as f:
        content = f.read()
        to_data = content.split('\n')
    with open(tst_file_location, 'r') as f:
        content = f.read()
        from_data = content.split('\n')
    for n, _ in enumerate(to_data[: -1]):
        print(30 * '_')
        print('>', from_data[n])
        print()
        print('Reply:', to_data[n])
