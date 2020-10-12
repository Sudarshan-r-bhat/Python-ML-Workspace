import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

data = pd.read_csv('./datasets/Ads_CTR_Optimisation_reinforcement.csv')


def random_selection():
    selected_in_each_row = list()
    reward = 0
    ads_col_frequency = [0] * data.shape[1]
    for x in data.values:
        r_col = random.randint(0, data.shape[1] - 1)
        selected_in_each_row.append(r_col)
        ads_col_frequency[r_col] += 1
        reward += x[r_col]
    print('random selection', ads_col_frequency)
    print('random selection', reward)
    plt.subplot(1, 2, 1)
    plt.hist(selected_in_each_row, orientation='horizontal', label='frequency of each ad ', color=['cyan'],
             histtype='barstacked', rwidth=0.9)
    plt.xlabel('frequency of selection')
    plt.ylabel('ad index')
    plt.title('ad selection based on random selection of ads')
    # plt.show()


random_selection()


selected_in_each_row = list()
ads_col_frequency = [0] * data.shape[1]
rewards_per_ad = [0] * data.shape[1]
total_reward = 0

for i in range(0, data.shape[0]):
    ad_index = 0
    max_upper_bound = 0
    for j in range(0, data.shape[1]):
        if ads_col_frequency[j] > 0:
            avg_reward = rewards_per_ad[j] / ads_col_frequency[j]
            delta_i = math.sqrt(3/2 * math.log(i + 1) / ads_col_frequency[j])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad_index = j
    selected_in_each_row.append(ad_index)
    ads_col_frequency[ad_index] += 1
    reward = data.values[i, ad_index]
    rewards_per_ad[ad_index] += reward
    total_reward += reward

print(ads_col_frequency)
print(total_reward)

# plot graph
plt.subplot(1, 2, 2)
plt.xlabel('frequency of selection')
plt.ylabel('ad index')
plt.title('ad selection based on UpperConfidenceBound')
plt.hist(selected_in_each_row, orientation='horizontal', label='frequency of each ad ', color=['cyan'],
         histtype='barstacked', rwidth=0.9)
plt.legend()
plt.show()




