from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm # recommended import according to the docs
from matplotlib.patches import Patch
from matplotlib import cm
colori=cm.get_cmap('tab10').colors

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

#for ML model
AL_stra='iGS'
n_queries=250
ss=4 #gsio

for metric in ['mae','rmse']:
        save_three_reg=[]
        save_three_reg_std=[]
        for regr_choosen in ['XGboost','SVR','RF','GP']:
            save_ave_across_users_for_shuffle=[]
            for shuffle in [13,34,42,70,104]:
                save_ave_across_users=[]
                for u in range(32):
                    for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                        main_path=regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'
                        user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(u) + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')[ss]  # [0] is ave of shuffle
                        save_ave_across_users.append(user_data.tolist())
                save_ave_across_users = save_ave_across_users
                save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
            save_three_reg.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
            save_three_reg_std.append(np.std(save_ave_across_users_for_shuffle,axis=0))

        main_path_for_save_fig = 'Plot_continuous_metrics_iGS'
        if not os.path.exists(main_path_for_save_fig):
            os.makedirs(main_path_for_save_fig)
        np.save(main_path_for_save_fig + '/' + metric + 'values', save_three_reg)
        #np.save(main_path_for_save_fig + '/' + metric + 'values'+flag_insca, save_three_reg_std)

        fig = plt.figure(figsize=(20, 10),dpi=100)
        leg = ['iQoE','iGS+SVR','iGS+RF','iGS+GP']
        conta = 0
        stile = ['-', '--', '-.', ':']
        col=['r',colori[2],colori[4],colori[0]]
        for regr in ['iQoE','SVR+iGS','RF+iGS','GP+iGS']:
            users_meanmean=save_three_reg[conta][20:]
            users_meanster=save_three_reg_std[conta][20:]
            #data1 = plt.scatter(range(n_queries+1-20), users_meanmean, marker='.',color=col[conta])
            #plt.errorbar(range(n_queries+1), users_meanmean, yerr=users_meanster, ls='none', color='k')#
            f = interp1d(range(n_queries+1-20), users_meanmean)
            plt.plot(range(n_queries+1-20), f(range(n_queries+1-20)), stile[conta], linewidth='7',color=col[conta])
            conta += 1
        #plt.grid()
        plt.xlabel("Number of SAs", fontdict=font_axes_titles)

        plt.xticks([0, 50-20, 100-20, 150-20, 200-20, 250-20], ['20', '50', '100', '150', '200', '250'])
        plt.ylabel(metric.upper(), fontdict=font_axes_titles)
        #plt.yticks(range(0, 20, 2))
        plt.gcf().subplots_adjust(bottom=0.2)  # add space down
        plt.yticks(np.arange(0, 13, 2))
        plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
        ax = plt.gca()
        ax.tick_params(axis='x', which='major', width=7, length=24)
        ax.tick_params(axis='y', which='major', width=7, length=24)
        ax.set_ylim([1.3, 12])
        #lege = [leg[-1], leg[1], leg[2], leg[3], leg[0]]
        #colorsi = [col[-1], col[1], col[2], col[3], col[0]]
        handles = [Patch(facecolor=color, label=label) for label, color in zip(leg, col)]
        #plt.legend(ncol=2,frameon=False, handles=handles, handlelength=2., handleheight=0.7, fontsize=40,bbox_to_anchor=(0.03, 0.07, 1, 1),handletextpad=0.1,columnspacing=0.5)
        #plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
        #plt.legend(ncol=3,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
        #plt.show()
        plt.savefig(main_path_for_save_fig + '/' + metric+'ml.pdf',bbox_inches='tight',)
        plt.close()








