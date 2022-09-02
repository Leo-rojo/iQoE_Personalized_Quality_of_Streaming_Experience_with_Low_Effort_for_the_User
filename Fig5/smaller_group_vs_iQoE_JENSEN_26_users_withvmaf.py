import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
colori=cm.get_cmap('tab10').colors
###############
q_fix=49
numbers_of_groups=[1, 2, 4, 8, 16, 32, 64, 128, 256]
font_axes_titles = {'family': 'sans-serif',
                'color':  'black',
                #'weight': 'bold',
                'size': 60,
                }
font_title = {'family': 'sans-serif',
        'color':  'black',
        #'weight': 'bold',
        'size': 60,
        }
font_general = {'family' : 'sans-serif',
        #'weight' : 'bold',
        'size'   : 60}
plt.rc('font', **font_general)
import os
mosgroup_26=np.load('./result_jensen/mae_each_query_worst_users_a.npy') #[9 groups][26 guys]
mosgroup_26_rmse=np.load('./result_jensen/rmse_each_query_worst_users_a.npy')
mosgroup_26_V=np.load('./result_jensen/mae_each_query_worst_users_V.npy') #[9 groups][26 guys]
mosgroup_26_rmse_V=np.load('./result_jensen/rmse_each_query_worst_users_V.npy')
iQoE_26_mae=np.load('./result_jensen/mae_'+str(q_fix)+'_worst_users_iqoe.npy') #[9 groups][26 guys]
iQoE_26_rmse=np.load('./result_jensen/rmse_'+str(q_fix)+'_worst_users_iqoe.npy')
avevamae=np.mean(mosgroup_26,axis=1)
avevarmse=np.mean(mosgroup_26_rmse,axis=1)
aveiqoemae=np.mean(iQoE_26_mae,axis=1)
aveiqoermse=np.mean(iQoE_26_rmse,axis=1)
avevamae_V=np.mean(mosgroup_26_V,axis=1)
avevarmse_V=np.mean(mosgroup_26_rmse_V,axis=1)
stdvamae=np.std(mosgroup_26,axis=1)
stdvarmse=np.std(mosgroup_26_rmse,axis=1)
stdvamae_V=np.std(mosgroup_26_V,axis=1)
stdvarmse_V=np.std(mosgroup_26_rmse_V,axis=1)
stdiqoemae=np.std(iQoE_26_mae,axis=1)
stdiqoermse=np.std(iQoE_26_rmse,axis=1)



fig = plt.figure(figsize=(20, 10), dpi=100)
barWidth=0.4
a = np.arange(0,13.5,1.5)
b = [i + barWidth-0.05 for i in a]
c = [i + barWidth-0.05 for i in b]
plt.bar(a, aveiqoemae.reshape(9),width = barWidth-0.1, color ='red',linewidth=0,label='iQoE',align='center',yerr=stdiqoemae.reshape(9))
plt.bar(b, avevamae_V,width = barWidth-0.1, color =colori[8],linewidth=0,label='V',align='center',yerr=stdvamae_V)
plt.bar(c, avevamae,width = barWidth-0.1, color =colori[4],linewidth=0,label='A',align='center',yerr=stdvamae)
plt.xticks(range(len(numbers_of_groups)),numbers_of_groups)
plt.xlabel("Number of reference groups",fontdict=font_axes_titles)
plt.ylabel('MAE', fontdict=font_axes_titles)


plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
dist=np.arange(0,13.5,1.5)
dist[-1]=12.1
plt.xticks(dist + barWidth-0.05, [1, 2, 4, 8, 16, 32, 64,128, 256])
plt.axhline(y=0, color='black')#plot iQoE various queries
# plt.axhline(y=maes_gsio[30], color='b', linestyle='-',label='iQoE_30q')#plot iQoE various queries
# plt.axhline(y=maes_gsio[50], color='r', linestyle='-',label='iQoE_50q')#plot iQoE various queries
# plt.axhline(y=maes_gsio[70], color='g', linestyle='-',label='iQoE_70q')#plot iQoE various queries
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
ax.set_ylim([0, 30])
plt.yticks(range(0, 35,5))
#ax.set_ylim([-2, 10])
# plt.title('Comparison nr chunks '+str(nr_chunk),fontdict=font_title)
#plt.legend(ncol=3, fontsize=60,frameon=False, bbox_to_anchor=(0.05, 0.1, 1., 1),labelspacing=0.1,handletextpad=0.1,columnspacing=0.5)
#plt.show()
plt.savefig('barplot_26users_groupvsiqoe_mae.pdf', bbox_inches='tight')
plt.close()
np.save('datamos',avevamae)
np.save('dataiqoe',aveiqoemae)
np.save('datamos_V',avevamae_V)

###rmse
fig = plt.figure(figsize=(20, 10), dpi=100)
barWidth=0.4
a = np.arange(0,13.5,1.5)
b = [i + barWidth-0.05 for i in a]
c = [i + barWidth-0.05 for i in b]
plt.bar(a, aveiqoermse.reshape(9),width = barWidth-0.1, color ='red',linewidth=0,label='iQoE',align='center',yerr=stdiqoermse.reshape(9))
plt.bar(b, avevarmse_V,width = barWidth-0.1, color =colori[8],linewidth=0,label='V',align='center',yerr=stdvarmse_V)
plt.bar(c, avevarmse,width = barWidth-0.1, color =colori[4],linewidth=0,label='A',align='center',yerr=stdvarmse)

plt.xticks(range(len(numbers_of_groups)),numbers_of_groups)
plt.xlabel("Number of reference groups",fontdict=font_axes_titles)
plt.ylabel('RMSE', fontdict=font_axes_titles)
ax.set_ylim([0, 30])

plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
dist=np.arange(0,13.5,1.5)
dist[-1]=12.1
plt.xticks(dist + barWidth-0.05, [1, 2, 4, 8, 16, 32, 64,128, 256])
plt.axhline(y=0, color='black')#plot iQoE various queries
# plt.axhline(y=maes_gsio[30], color='b', linestyle='-',label='iQoE_30q')#plot iQoE various queries
# plt.axhline(y=maes_gsio[50], color='r', linestyle='-',label='iQoE_50q')#plot iQoE various queries
# plt.axhline(y=maes_gsio[70], color='g', linestyle='-',label='iQoE_70q')#plot iQoE various queries
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
ax.set_ylim([0, 30])
plt.yticks(range(0, 35,5))
#ax.set_ylim([-2, 10])
# plt.title('Comparison nr chunks '+str(nr_chunk),fontdict=font_title)
#plt.legend(ncol=3,fontsize=60,frameon=False,labelspacing=0.1,handletextpad=0.1,bbox_to_anchor=(0.05, 0.1, 1., 1),columnspacing=0.5)
#plt.show()
plt.savefig('barplot_26users_groupvsiqoe_rmse.pdf', bbox_inches='tight')
plt.close()
np.save('datamos_rmse',avevarmse)
np.save('dataiqoe_rmse',aveiqoermse)
np.save('datamos_rmse_V',avevarmse_V)

print('done')