import csv, math, random
import statistics as st


# here we are only loading and converting all the numerical lines of data from string to float.
def load_csv(filename):
    lines = csv.reader(open(filename, 'r'))
    dataSet = list(lines)
    for i in range(len(dataSet)):
        dataSet[i] = [float(x) for x in dataSet[i]]
    return dataSet


# we are going to pick random instances for testing from the dataset.
# steps followed are: store dataset into trainSet and pop elements from trainSet w.r.t
# randomly generated index. and append to testSet
def splitDataSet(dataSet, splitRatio):
    testSize = int(len(dataSet) * splitRatio)
    trainSet = list(dataSet)
    testSet = []
    while len(testSet) < testSize:
        index = random.randrange(len(trainSet))
        testSet.append(trainSet.pop(index))
    return [trainSet, testSet]


def summarizeByClass(trainingSet):
    separated = separateByClass(trainingSet)
    summary = dict() # used to store mean and std.deviation of 2 classes/ pos, neg instances.
    for classValue, instances in separated.items():
        summary[classValue] = compute_mean_std_values(instances)
    return summary


# we are dividing the dataset into two classes (for values 1 and 0).
def separateByClass(trainingSet):
    separated = dict()
    for i in range(len(trainingSet)):
        x = trainingSet[i]
        if x[-1] not in separated:
            separated[x[-1]] = []
        separated[x[-1]].append(x)
    return separated


# zip is used to list the first to last occurrences in the tuples. in other words it transposes the 2-D matrix.
# the functions calculates mean and stdev of all the columns
def compute_mean_std_values(instances): # instances == dataset. pass by reference is a must.
    mean_std = [
        (st.mean(attribute), st.stdev(attribute))
        for attribute in zip(*instances)  # here zip(*instances) will transpose the matrix.
    ]
    del mean_std[-1] # we are excluding the label they say.?
    return mean_std


def performClassification(summaries, testSet):
    predictions = [] # this is a local variable, it is not affecting the global variable.
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def predict(summaries, testVector):
    all_probabilities = calcClassProb(summaries, testVector)
    bestLabel, bestProb = None, -1
    for label, prob in all_probabilities.items():
        if bestLabel is None or prob > bestProb:
            bestLabel = label
            bestProb = prob
    return bestLabel


def calcClassProb(summaries, testVector):
    prob = dict()
    for classValue, classSummaries in summaries.items():
        prob[classValue] = 1  # initial value.
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = testVector[i]
            prob[classValue] *= estimatedProbability(x, mean, stdev)
    return prob


def estimatedProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    final_accuracy = (correct / float(len(testSet))) * 100.0
    return final_accuracy


dataSet = load_csv('dataSet5.csv')
print('Pima Indian Diabetes Dataset has been loaded')
print('Total instances available are ', len(dataSet))
print('Total attributes/columns present are ', len(dataSet[0]) - 1)
splitRatio = 0.3 # are splitting our dataSets for Training and Testing purpose.

trainingSet, testSet = splitDataSet(dataSet, splitRatio)
print('training lines = {0}\ntesting lines = {1}'.format(len(trainingSet), len(testSet)))
# we will summarize the training set.
summarize = summarizeByClass(trainingSet)
# next we shall make predictions
predictions = performClassification(summarize, testSet)
accuracy = getAccuracy(testSet, predictions) # we cross verify our predictions with testSet.
print('Naive Bayes classifier is {0}% accurate'.format(accuracy))

