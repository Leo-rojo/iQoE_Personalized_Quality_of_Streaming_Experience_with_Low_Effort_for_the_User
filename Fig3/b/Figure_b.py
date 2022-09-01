import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.patches as mpatches
import matplotlib.font_manager as font_manager
import os
#need to be in folder b

#take mos scores
xls = pd.ExcelFile(r"../hdtv_scores.xlsx") #use r before absolute file path
sheetX = xls.parse(0)
mosarray=sheetX['mos'].tolist()

#takes user scores and order them based on hightest discrepancy
collect_all=[]
users_scores=np.load('../users_scores_hdtv.npy')
right_order=[26	,8	,22	,31	,12	,16	,28	,11	,7	,30	,24	,10	,21	,5	,13	,15	,18	,2	,4	,3	,9	,17	,14	,23	,25	,20	,27	,19	,6	,32	,1	,29]

#sort users scores
ordered_users=[]
for num in right_order:
    ordered_users.append(users_scores[num - 1].tolist())

#ECDF for 4users and average user
lab=['HA','H1','H32','H2','H31']
my_pal = ["#4b8521", "b", "c",'grey','m']
fig=plt.figure(figsize=(20, 10),dpi=100)
subsec=[mosarray,ordered_users[0],ordered_users[-1],ordered_users[1],ordered_users[-2]]

#plot ecdf
for i in range(5):
    ecdf = sm.distributions.ECDF(subsec[i])
    plt.step(ecdf.x, ecdf.y,label=lab[i],color=my_pal[i],linewidth=6.0)
plt.margins(0,0)
plt.tick_params(axis='both',which='major', labelsize=20, width=3.5, length=20)
plt.ylabel('Fraction of scores', fontsize=55)
plt.xticks(fontsize=55)
plt.yticks(fontsize=55)
plt.xlabel('SA score', fontsize=55)
plt.xticks([1,20,40,60,80,100])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['0','0.2','0.4','0.6','0.8','1'])
HA_patch = mpatches.Patch(color='#4b8521', label='HA')
H1_patch = mpatches.Patch(color='b', label='H1')
H32_patch = mpatches.Patch(color='c', label='H32')
H2_patch = mpatches.Patch(color='grey', label='H2')
H31_patch = mpatches.Patch(color='m', label='H31')
font = font_manager.FontProperties(size=50)
plt.legend(handles=[H1_patch,H2_patch,H31_patch,H32_patch,HA_patch],loc='upper center', bbox_to_anchor=(0.5, 1.23), fontsize=50, ncol=5, framealpha=0, columnspacing=0.5,handletextpad=0.4,prop=font)
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('Figure_b.pdf',bbox_inches='tight')