from scipy.interpolate import interp1d
import statsmodels.api as sm # recommended import according to the docs
#for ML model
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
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

n_queries=250
leg = ['RS+XGB', 'CU+XGB', 'GS+XGB', 'QBC+XGB', 'iQoE']
colors=[colori[0],colori[1],colori[2],colori[4],'r']
regr_choosen='XGboost'

for metric in ['mae','rmse']:
        save_five_ss = []
        for ss in range(5):
            save_ave_across_users_for_shuffle = []
            for shuffle in [13, 34, 42, 70, 104]:
                save_ave_across_users = []
                for u in range(32):
                    for QoE_model in ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']:
                        main_path = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'
                        user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(
                            u) + '/' + 'shuffle_' + str(shuffle) + '/' + metric + '/scores_for_ALstrat.npy')[
                            ss]  # [0] is ave of shuffle
                        save_ave_across_users.append(user_data.tolist())
                save_ave_across_users_for_shuffle.append(save_ave_across_users)
            save_five_ss.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
        #np.save('values'+metric+totuser,save_five_ss)


        fix_query=49

        main_path_for_save_fig = 'Plot_ECDF'
        if not os.path.exists(main_path_for_save_fig):
            os.makedirs(main_path_for_save_fig)
        #np.save(main_path_for_save_fig + '/' + metric + 'values_ave', final_ave)
        #np.save(main_path_for_save_fig + '/' + metric + 'values_std', final_std)
        save_distributions=[]
        fig = plt.figure(figsize=(20, 10), dpi=100)
        conta = 0
        stile = ['--', ':', '--', '-.', '-']
        for ss in range(5):

            final_ave_elab=[save_five_ss[ss][i][fix_query] for i in range(256)]
            ecdf = sm.distributions.ECDF(final_ave_elab)
            save_distributions.append(ecdf)
            plt.step(ecdf.x, ecdf.y, label=leg[ss], linewidth=7.0, color=colors[ss], linestyle=stile[conta])
            conta+=1
        np.save('Plot_ECDF/'+'values_ecdf'+metric,save_distributions)

        plt.xlabel(metric.upper(), fontdict=font_axes_titles)
        plt.ylabel('Fraction of users', fontdict=font_axes_titles)
        #plt.xticks(np.arange(0,20,5))
        # plt.title('ECDF',fontdict=font_title)
        lege = [leg[-1], leg[1], leg[2], leg[3], leg[0]]
        colorsi = [colors[-1], colors[1], colors[2], colors[3], colors[0]]
        handles = [
            Patch(facecolor=color, label=label)
            for label, color in zip(lege, colorsi)
        ]
        #plt.legend(ncol=2, frameon=False, handles=handles, handlelength=2., loc='lower right',handleheight=0.7,fontsize=40,handletextpad=0.1,columnspacing=0.5)
        plt.gcf().subplots_adjust(bottom=0.2)  # add space down
        plt.gcf().subplots_adjust(left=0.15)  # add space left
        plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
        ax = plt.gca()
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
        ax.tick_params(axis='x', which='major', width=7, length=24)
        ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
        ax.set_xlim([0, 20])
        # plt.xlim(0, 25)
        # plt.show()
        plt.savefig('Plot_ECDF/'+metric+'_ECDF.pdf', bbox_inches='tight')
        plt.close()







