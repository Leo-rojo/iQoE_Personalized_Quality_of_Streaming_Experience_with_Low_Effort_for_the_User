import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

dir_path=('./time_over')
res=[]
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)
l=[]
for i in res:
    y=i.split('.')
    l.append(y[0].split('_')[2])

print(l)



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

mod_names=['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']
bymod_mean=[]
bymod_std=[]
allu = []
for mod in mod_names:
    for u in range(32):
        eachu=[]
        eachu.append(os.path.getsize("./mq/" +str(u)+mod+'m_q' + 'initial'+'.json') / 1024)
        for nq in range(250):
            eachu.append(os.path.getsize("./mq/"+str(u)+mod+'m_q'+str(nq)+'.json')/1024)
        allu.append(eachu)

bymod_mean.append(np.mean(allu,axis=0))
bymod_std.append(np.std(allu,axis=0))
np.save('space_ave',bymod_mean)
np.save('space_std',bymod_std)

n_queries=250
fig = plt.figure(figsize=(20, 10),dpi=100)
c=0
error_freq=np.arange(1,250,10)
for i in bymod_mean:
    plt.plot(i,label=mod_names[c],color='r',linewidth='7')
    plt.fill_between(np.arange(len(i)), i - bymod_std[c], i + bymod_std[c],alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    # data1 = plt.scatter(range(n_queries + 1), i, marker='.', color='r')
    # plt.errorbar(error_freq,np.take(i,error_freq), yerr=np.take(bymod_std[c],error_freq), ls='none', color='k')#
    # f = interp1d(range(n_queries + 1), i)
    # plt.plot(range(n_queries + 1), f(range(n_queries + 1)), '-', linewidth='2',  color='r')#label=leg[conta],
    c+=1

plt.xlabel("Iteration number", fontdict=font_axes_titles)
plt.xticks([0, 50, 100, 150, 200, 250], ['1', '50', '100', '150', '200', '250'])
plt.ylabel('Memory, KB', fontdict=font_axes_titles)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
plt.yticks(range(0,500,100))
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
ax.set_ylim([0, 400])
#plt.show()
#plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
#plt.legend(ncol=5,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
#plt.legend(ncol=2,fontsize=40)
plt.savefig('./space_all.pdf',bbox_inches='tight')

bymod_mean_time=[]
bymod_std_time=[]
foru = []
for mod in mod_names:
    for u in range(32):
        #foru.append(np.cumsum(np.load('./Regression/time_over/time_overhead_'+str(u)+mod+'.npy')))
        foru.append(np.load('./time_over/time_overhead_'+str(u)+mod+'.npy'))
bymod_mean_time.append(np.mean(foru, axis=0))
bymod_std_time.append(np.std(foru, axis=0))
np.save('time_ave',bymod_mean_time)
np.save('time_std',bymod_std_time)
fig1 = plt.figure(figsize=(20, 10),dpi=100)
c=0
for i in bymod_mean_time:
    plt.plot(i,label=mod_names[c],color='r',linewidth='7')
    plt.fill_between(np.arange(len(i)), i - bymod_std_time[c], i + bymod_std_time[c], alpha=0.5, edgecolor='#CC4F1B',
                     facecolor='#FF9848')
    c+=1
plt.xlabel("Iteration number", fontdict=font_axes_titles)
plt.xticks([0, 50, 100, 150, 200, 250], ['1', '50', '100', '150', '200', '250'])
plt.ylabel('Time, s', fontdict=font_axes_titles)
plt.gcf().subplots_adjust(bottom=0.2)  # add space down
plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
plt.yticks(np.arange(0,1,0.2),['0','0.2','0.4','0.6','0.8'])
ax = plt.gca()
ax.tick_params(axis='x', which='major', width=7, length=24)
ax.tick_params(axis='y', which='major', width=7, length=24)
ax.set_ylim([0, 0.95])

#plt.show()
#plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
#plt.legend(ncol=5,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
#plt.legend(ncol=2,fontsize=40)
plt.savefig('./time_all.pdf',bbox_inches='tight')
