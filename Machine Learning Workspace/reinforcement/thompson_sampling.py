import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


data = pd.read_csv('./datasets/Ads_CTR_Optimisation_reinforcement.csv')

col_selected_in_each_row = list()
frequency_rewards_1 = [0] * data.shape[1]
frequency_rewards_0 = [0] * data.shape[1]
total_reward = 0

for i in range(0, data.shape[0]):
    ad_index = 0
    max_random = 0
    for j in range(0, data.shape[1]):
        random_beta = random.betavariate(frequency_rewards_1[j] + 1, frequency_rewards_0[j] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad_index = j

    col_selected_in_each_row.append(ad_index)
    reward = data.values[i, ad_index]
    total_reward += reward
    if reward == 1:
        frequency_rewards_1[ad_index] += 1
    else:
        frequency_rewards_0[ad_index] += 1


print(total_reward)

# plot graph
plt.xlabel('frequency of selection')
plt.ylabel('ad index')
plt.title('ad selection based on Thompson Sampling')
plt.hist(col_selected_in_each_row, orientation='horizontal', label='frequency of each ad ', color=['cyan'],
         histtype='barstacked', rwidth=0.9)
plt.xticks(np.linspace(0, 10000, 11))
plt.legend()
plt.show()




