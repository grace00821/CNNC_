import numpy as np

y = np.load('predict_results_no_y_1/y_predict.npy')

import pandas as pd
df_obs = pd.read_csv('data/mmukegg_new_new_unique_rand_labelx.txt', sep='\t', header=None)

original = []
index = []

## Organize original data according to the regulator
## [['regulator1', label0, label1, etc], ['regulator2', label0, label1, etc], etc]
for i in range(len(df_obs[0])):
    check = 0
    reg = df_obs[0][i]
    if len(original) != 0:
        for j in range(len(original)):
            if original[j][0] == reg:
                check = 1
                original[j].append(df_obs[2][i])
                index[j].append(i)
    if check == 0:
        temp1 = [reg, df_obs[2][i]]
        temp2 = [reg, i]
        original.append(temp1)
        index.append(temp2)

## Classify original data so it is comparable to the predicted data
## [['regulator1', [0,0,1], [0,1,0], etc], ['regulator2', [1,0,0], [0,1,0], etc], etc]
## no regulator name removed
from sklearn.preprocessing import label_binarize

binOriginal = []
for i in range(len(original)):
    temp = original[i][1:]
    temp2 = label_binarize(temp, classes=[0,1,2])
    binOriginal.append(temp2)
# # Per regulator
# print(binOriginal[0].shape)
# print(type(binOriginal[0]))
# print(type(binOriginal[0][0]))
# print(binOriginal[0])
# # Per each result
# print(binOriginal[0][0])
# # Per each label
# print(binOriginal[0][0][0])

## Sort predicted data according to regulator
pdata = []
for i in range(len(index)):
    temp = []
    for j in range(len(index[i])):
        if j != 0:
            if index[i][j] >= len(y):
                break
            temp.append(y[index[i][j]])
    if len(temp) != 0:
        pdata.append(np.array(temp))
# # Per regulator
# print(type(pdata[0]))
# print(type(pdata[0][0]))
# # Per each result
# print(pdata[0][0])
# # Per each label
# print(pdata[0][0][0])


## Generate ROC_AUC
from sklearn import metrics
fpr = []
tpr = []
roc_auc = []
for i in range(len(binOriginal)):
    roc_auc_temp = []
    fpr_temp = []
    tpr_temp = []
    origin_temp = binOriginal[i]
    predic_temp = pdata[i]
    for j in range(3):
        ftemp, ttemp, _ = metrics.roc_curve(origin_temp[:,j], predic_temp[:,j])
        fpr_temp.append(ftemp)
        tpr_temp.append(ttemp)
        roc_auc_temp.append(metrics.auc(fpr_temp[j], tpr_temp[j]))
    roc_auc.append(roc_auc_temp)
    fpr.append(np.array(fpr_temp))
    tpr.append(np.array(tpr_temp))

roc_auc = np.array(roc_auc)
# fpr = np.array(fpr)
# print(fpr.shape)
# tpr = np.array(tpr)
# print(tpr.shape)

print('original:\n', binOriginal[6])
print('predict:\n', pdata[6])
print('\n')
print('roc_auc:\n', roc_auc[6])
print('fpr:\n', fpr[6])
print('tpr:\n', tpr[6])

# np.save("roc_auc.npy", roc_auc)
# np.save("fpr.npy", fpr)
# np.save("tpr.npy", tpr)

## Label 0
# import matplotlib.pyplot as plt

# fig = plt.figure()
# label0 = roc_auc[:,0]
# median = np.nanmedian(label0)
# idx = np.nanargmin(np.abs(label0-median))
# for i in range(len(fpr)):
#     if i == idx:
#         plt.plot(fpr[i][0], tpr[i][0], label='AUC=%.4f' %(median), linewidth=3, color='r', zorder=2)
#     else:
#         plt.plot(fpr[i][0], tpr[i][0], linewidth=1, color='#C5C5C5', zorder=1)
# plt.legend()
# plt.savefig('ROC_curve0.png')

# ## Label 1
# fig = plt.figure()
# label1 = roc_auc[:,1]
# median = np.nanmedian(label1)
# idx = np.nanargmin(np.abs(label1-median))
# for i in range(len(fpr)):
#     if i == idx:
#         plt.plot(fpr[i][1], tpr[i][1], label='AUC=%.4f' %(median), linewidth=3, color='r', zorder=2)
#     else:
#         plt.plot(fpr[i][1], tpr[i][1], linewidth=1, color='#C5C5C5', zorder=1)
# plt.legend()
# plt.savefig('ROC_curve1.png')

# ## Label 2
# fig = plt.figure()
# label2 = roc_auc[:,2]
# median = np.nanmedian(label2)
# idx = np.nanargmin(np.abs(label2-median))
# for i in range(len(fpr)):
#     if i == idx:
#         plt.plot(fpr[i][2], tpr[i][2], label='AUC=%.4f' %(median), linewidth=3, color='r', zorder=2)
#     else:
#         plt.plot(fpr[i][2], tpr[i][2], linewidth=1, color='#C5C5C5', zorder=1)
# plt.legend()
# plt.savefig('ROC_curve2.png')