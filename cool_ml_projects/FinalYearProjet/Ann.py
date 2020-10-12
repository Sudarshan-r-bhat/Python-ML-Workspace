import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

data_attack = pd.read_csv('./datasets/dataset_attack.csv')
data_normal = pd.read_csv('./datasets/dataset_normal.csv')

data_normal.columns = ['frame.len', 'frame.protocols', 'ip.hdr_len',
                       'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
                       'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
                       'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
                       'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
                       'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
                       'tcp.time_delta', 'class']
data_attack.columns = ['frame.len', 'frame.protocols', 'ip.hdr_len',
                       'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
                       'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport',
                       'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
                       'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
                       'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
                       'tcp.time_delta', 'class']

data_normal = data_normal.drop(['ip.src', 'ip.dst', 'frame.protocols'], axis=1)
data_attack = data_attack.drop(['ip.src', 'ip.dst', 'frame.protocols'], axis=1)

features = ['frame.len', 'ip.hdr_len',
            'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset',
            'ip.ttl', 'ip.proto', 'tcp.srcport', 'tcp.dstport',
            'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr',
            'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
            'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size',
            'tcp.time_delta']

X_normal = data_normal[features].values
X_attack = data_attack[features].values
Y_normal = data_normal['class']
Y_attack = data_attack['class']
X = np.concatenate((X_normal, X_attack))
Y = np.concatenate((Y_normal, Y_attack))

scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
scalar.fit(X)
X = scalar.transform(X)

for i in range(0, len(Y)):
    if Y[i] == "attack":
        Y[i] = 0
    else:
        Y[i] = 1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(Y_train)

### realtime train

model = Sequential()
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.save("./models/ann_model.h5")

# model = load_model('ann_model.h5')

history = model.fit(X_train, Y_train, validation_split=0.33, epochs=100)
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()


predict = model.predict(X_test, verbose=1)

tp = 0
tn = 0
fp = 0
fn = 0
predictn = predict.flatten().round()
predictn = predictn.tolist()
Y_testn = Y_test.tolist()
for i in range(len(Y_testn)):
    if predictn[i] == 1 and Y_testn[i] == 1:
        tp += 1
    elif predictn[i] == 0 and Y_testn[i] == 0:
        tn += 1
    elif predictn[i] == 0 and Y_testn[i] == 1:
        fp += 1
    elif predictn[i] == 1 and Y_testn[i] == 0:
        fn += 1
to_heat_map = [[tn, fp], [fn, tp]]
to_heat_map = pd.DataFrame(to_heat_map, index=["Attack", "Normal"], columns=["Attack", "Normal"])
ax = sns.heatmap(to_heat_map, annot=True, fmt="d")
plt.show()

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

