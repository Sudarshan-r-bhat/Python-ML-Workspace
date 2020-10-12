# chat bot Sentdex tutorials.
# we will be using datasets from reddit.
import sqlite3
from datetime import datetime
import json

time_frame = '2015-05'
sql_transaction = []
connection = sqlite3.connect('{}.db'.format(time_frame)) # we will connect to the database named 2015-05
c = connection.cursor() # we shall define the pointer


def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply"
              "(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, "
              "unix INT , score INT)")


def data_format(data):
    data = data.replace('\n', 'newlinechar').replace('\r', 'newlinechar').replace('"', "'")
    return data


def find_parent(pid):
    try:
        sql = "SELECT comment FROM parent_reply where comment_id = {} LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        #print('find parent', e)
        return False


def find_existing_score(pid):
    try:
        sql = "SELECT score FROM parent_reply where parent_id = {} LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result is not None:
            return result[0]
        else:
            return False
    except Exception as e:
        #print('find parent', e)
        return False

def acceptable(data): # if the comment data have more than the required values, we are not going to consider them.
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]' or data == '[removed]'
        return False
    else:
        return True


def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? 
        WHERE parent_id =?;""".format(parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception e:
        print('s-update insertion ', str(e))


def sql_insert_has_parent(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) 
        VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception e:
        print('s-parent insertion ', str(e))


def sql_insert_no_parent(commentid, parentid, parent, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score)
         VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception e:
        print('s-no-parent insertion ', str(e))


def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
            connection.commit()
            sql_transaction = []

if __name__ == '__main__()':
    create_table()
    row_counter = 0
    paired_rows = 0
    with open('D:/workstation/datasets/chatdata/reddit_data/{}/RC_{}'.format(time_frame.split('-')[0],
                                                                            time_frame), buffering=1000) as f:
        for row in f:
            row_counter += 1
            row = json.loads(row)
            parent_id = row['parent_id']
            body = format_data(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            subreddit = row['subreddit']
            comment_id = row['name']
            parent_data = find_parent(parent_id)
            if score >= 2:
                if acceptable(body):
                    existing_comment_score = find_existing_score(parent_id)
                    if existing_comment_score: # if there is atleast one comment selected.
                        if score > existing_comment_score:
                            sql_insert_replace_comment(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                    else:
                        if parent_data:
                            sql_insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_utc, score)
                        else:
                            sql_insert_no_parent(comment_id, parent_id, body, subreddit, created_utc, score)
            if row_counter % 100000 == 0:
                print('total rows read: {}, paired rows:{}, time: {}, '.format(row_counter, paired_rows, str(datetime.now())))
