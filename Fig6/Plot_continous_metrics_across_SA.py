import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
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

n_queries=250
regr_choosen='XGboost'


for metric in ['mae','rmse']:
        save_five_ss=[]
        save_five_ss_stdv=[]
        for ss in range(5):
            save_ave_across_users_for_shuffle=[]
            for shuffle in [13,34,42,70,104]:
                save_ave_across_users=[]
                for u in range(32):
                    for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                        main_path = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'
                        user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(
                            u) + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')[ss]  # [0] is ave of shuffle
                        save_ave_across_users.append(user_data.tolist())
                save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
            save_five_ss.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
            save_five_ss_stdv.append(np.std(save_ave_across_users_for_shuffle,axis=0))
        main_path_for_save_fig = 'Plot_continuous_metrics_' + regr_choosen
        if not os.path.exists(main_path_for_save_fig):
            os.makedirs(main_path_for_save_fig)
        np.save(main_path_for_save_fig+'/'+metric+'values',save_five_ss)
        #np.save(main_path_for_save_fig + '/' + metric+'values'+flag_insca, save_five_ss_stdv)

        fig = plt.figure(figsize=(20, 10),dpi=100)
        leg = ['RS+XGB', 'CU+XGB', 'GS+XGB', 'QBC+XGB', 'iQoE']
        colors=[colori[0],colori[1],colori[2],colori[4],'r']
        stile=['-',':','--','-.','-']
        conta = 0
        for kind_strategy in leg:
            kind_strategy_idx=leg.index(kind_strategy)
            users_meanmean=save_five_ss[conta][20:]
            users_meanster=save_five_ss_stdv[conta][20:]
            #data1 = plt.scatter(range(n_queries+1-20), users_meanmean, marker='.',color=colors[conta] )
            #plt.errorbar(range(n_queries+1), users_meanmean, yerr=users_meanster, ls='none', color='k')#
            f = interp1d(range(n_queries+1-20), users_meanmean)
            plt.plot(range(n_queries+1-20), f(range(n_queries+1-20)), stile[conta], linewidth='7', label=leg[conta],color=colors[conta])
            conta += 1
        #plt.grid()
        plt.xlabel("Number of SAs", fontdict=font_axes_titles)
        plt.xticks([0, 50-20, 100-20, 150-20, 200-20, 250-20], ['20', '50', '100', '150', '200', '250'])
        plt.ylabel(metric.upper(), fontdict=font_axes_titles)

        plt.yticks(np.arange(0,11,2))
        plt.gcf().subplots_adjust(bottom=0.2)  # add space down
        plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
        ax = plt.gca()
        ax.tick_params(axis='x', which='major', width=7, length=24)
        ax.tick_params(axis='y', which='major', width=7, length=24)
        ax.set_ylim([1.3, 10])
        #plt.show()
        #plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
        #plt.legend(ncol=5,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
        lege=[leg[-1],leg[1],leg[2],leg[3],leg[0]]
        colorsi=[colors[-1],colors[1],colors[2],colors[3],colors[0]]
        handles = [Patch(facecolor=color, label=label) for label, color in zip(lege, colorsi)]
        #plt.legend(ncol=2,frameon=False, handles=handles, handlelength=2., handleheight=0.7, fontsize=40,bbox_to_anchor=(0.03, 0.07, 1, 1),handletextpad=0.1,columnspacing=0.5)
        #plt.legend(fontsize=30)
        #plt.show()
        plt.savefig(main_path_for_save_fig + '/' + metric+'.pdf',bbox_inches='tight')
        plt.close()





