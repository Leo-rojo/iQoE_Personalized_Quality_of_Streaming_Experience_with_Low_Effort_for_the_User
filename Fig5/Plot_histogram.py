import matplotlib.pyplot as plt
import numpy as np
q_fix=49 #nr of SA considered
numbers_of_groups=[1, 2, 4, 8, 16, 32, 64, 128, 256]

font_axes_titles = {'family': 'sans-serif',
                'color':  'black',
                'weight': 'bold',
                'size': 50,
                }
font_title = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'bold',
        'size': 50,
        }
font_general = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 50}
plt.rc('font', **font_general)
import os
mosgroup_26=np.load('./result_jensen/mae_each_query_worst_users_a.npy') #[9 groups][26 guys]
mosgroup_26_rmse=np.load('./result_jensen/rmse_each_query_worst_users_a.npy')
iQoE_26_mae=np.load('./result_jensen/mae_'+str(q_fix)+'_worst_users_iqoe.npy') #[9 groups][26 guys]
iQoE_26_rmse=np.load('./result_jensen/rmse_'+str(q_fix)+'_worst_users_iqoe.npy')
avevamae=np.mean(mosgroup_26,axis=1)
avevarmse=np.mean(mosgroup_26_rmse,axis=1)
aveiqoemae=np.mean(iQoE_26_mae,axis=1)
aveiqoermse=np.mean(iQoE_26_rmse,axis=1)
stdvamae=np.std(mosgroup_26,axis=1)
stdvarmse=np.std(mosgroup_26_rmse,axis=1)
stdiqoemae=np.std(iQoE_26_mae,axis=1)
stdiqoermse=np.std(iQoE_26_rmse,axis=1)



fig = plt.figure(figsize=(20, 10), dpi=100)
barWidth=0.4
a = range(len(avevamae))
b = [i + barWidth-0.05 for i in a]
plt.bar(a, avevamae,width = barWidth-0.1, color ='grey',linewidth=0,label='Group-QoE',align='center',yerr=stdvamae)
plt.bar(b, aveiqoemae.reshape(9),width = barWidth-0.1, color ='red',linewidth=0,label='iQoE 50q',align='center',yerr=stdiqoemae.reshape(9))
plt.xticks(range(len(numbers_of_groups)),numbers_of_groups)
plt.xlabel("Nr of groups",fontdict=font_axes_titles)
plt.ylabel('MAE', fontdict=font_axes_titles)
plt.yticks(range(0, 40,5))
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
plt.xticks(np.arange(9) + barWidth-0.23, [1, 2, 4, 8, 16, 32, 64,128, 256])
plt.axhline(y=0, color='black')#plot iQoE various queries
# plt.axhline(y=maes_iGS[30], color='b', linestyle='-',label='iQoE_30q')#plot iQoE various queries
# plt.axhline(y=maes_iGS[50], color='r', linestyle='-',label='iQoE_50q')#plot iQoE various queries
# plt.axhline(y=maes_iGS[70], color='g', linestyle='-',label='iQoE_70q')#plot iQoE various queries
ax = plt.gca()
#ax.set_ylim([-2, 10])
# plt.title('Comparison nr chunks '+str(nr_chunk),fontdict=font_title)
plt.legend(fontsize=40,frameon=False)
#plt.show()
plt.savefig('barplot_26users_groupvsiqoe_mae.pdf', bbox_inches='tight')
plt.close()
np.save('datamos',avevamae)
np.save('dataiqoe',aveiqoemae)

###rmse
fig = plt.figure(figsize=(20, 10), dpi=100)
barWidth=0.4
a = range(len(avevarmse))
b = [i + barWidth-0.05 for i in a]
plt.bar(a, avevarmse,width = barWidth-0.1, color ='grey',linewidth=0,label='Group-QoE',align='center',yerr=stdvarmse)
plt.bar(b, aveiqoermse.reshape(9),width = barWidth-0.1, color ='red',linewidth=0,label='iQoE 50q',align='center',yerr=stdiqoermse.reshape(9))
plt.xticks(range(len(numbers_of_groups)),numbers_of_groups)
plt.xlabel("Nr of groups",fontdict=font_axes_titles)
plt.ylabel('RMSE', fontdict=font_axes_titles)
plt.yticks(range(0, 40,5))
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.gcf().subplots_adjust(left=0.15)  # add space left
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
plt.xticks(np.arange(9) + barWidth-0.23, [1, 2, 4, 8, 16, 32, 64, 128, 256])
plt.axhline(y=0, color='black')#plot iQoE various queries
# plt.axhline(y=maes_iGS[30], color='b', linestyle='-',label='iQoE_30q')#plot iQoE various queries
# plt.axhline(y=maes_iGS[50], color='r', linestyle='-',label='iQoE_50q')#plot iQoE various queries
# plt.axhline(y=maes_iGS[70], color='g', linestyle='-',label='iQoE_70q')#plot iQoE various queries
ax = plt.gca()
#ax.set_ylim([-2, 10])
# plt.title('Comparison nr chunks '+str(nr_chunk),fontdict=font_title)
plt.legend(fontsize=40,frameon=False)
#plt.show()
plt.savefig('barplot_26users_groupvsiqoe_rmse.pdf', bbox_inches='tight')
plt.close()
np.save('datamos_rmse',avevarmse)
np.save('dataiqoe_rmse',aveiqoermse)

print('done')