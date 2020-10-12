import csv
dataList = []
with open('dataset10.0.txt', 'r') as fp:
    line = fp.readline()
    while line:
        dataList.append(line[:-1].split(' '))
        line = fp.readline()
    fp.close()
print(dataList)
with open('dataset10.csv', 'w') as fp:
    wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
    for x in dataList:
        wr.writerow(x)
    fp.close()
# note you will have to remove the extra lines by yourself.
