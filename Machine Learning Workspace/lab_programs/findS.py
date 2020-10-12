import csv

with open('dataset1.csv', 'r') as fp:
    reader = csv.reader(fp)
    reader = list(reader)

result = reader[0]  # this is the first positive instance.
for i in range(len(reader)):
    if reader[i][-1].upper() == 'YES':
        for j, x in enumerate(reader[i]):
            if x != result[j] and result[j] != '?':
                result[j] = '?'
print('after generalization: ', result[:-1])
