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

leg=['iQoE','iGS+SVR','iGS+RF','iGS+GP']
AL_stra='iGS'
n_queries=250
ss=4 #iGS

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

                save_ave_across_users_for_shuffle.append(save_ave_across_users)
            save_three_reg.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
            save_three_reg_std.append(np.std(save_ave_across_users_for_shuffle,axis=0))
        #np.save('values'+metric+totuser,save_three_reg)


        fix_query=49
        col=['r',colori[2],colori[4],colori[0]]
        main_path_for_save_fig = 'Plot_ECDF'
        if not os.path.exists(main_path_for_save_fig):
            os.makedirs(main_path_for_save_fig)
        #np.save(main_path_for_save_fig + '/' + metric + 'values_ave', final_ave)
        #np.save(main_path_for_save_fig + '/' + metric + 'values_std', final_std)
        fig = plt.figure(figsize=(20, 10), dpi=100)
        save_distributions=[]
        conta = 0
        stile = ['-', '--', '-.', ':']
        for kind_reg in range(4):
            final_ave_elab=[save_three_reg[kind_reg][i][fix_query] for i in range(256)]
            ecdf = sm.distributions.ECDF(final_ave_elab)
            save_distributions.append(ecdf)
            plt.step(ecdf.x, ecdf.y, label=leg[kind_reg], linewidth=7.0, color=col[kind_reg],linestyle=stile[conta])
            conta+=1
        np.save('Plot_ECDF/values' + metric, save_distributions)

        plt.xlabel(metric.upper(), fontdict=font_axes_titles)
        plt.ylabel('Fraction of users', fontdict=font_axes_titles)
        #plt.xticks(np.arange(0,20,5))
        # plt.title('ECDF',fontdict=font_title)
        handles = [
            Patch(facecolor=color, label=label)
            for label, color in zip(leg, col)
        ]
        #plt.legend(ncol=2, frameon=False, handles=handles, handlelength=2., loc='lower right', handleheight=0.7,fontsize=40, handletextpad=0.1, columnspacing=0.5)
        plt.gcf().subplots_adjust(bottom=0.2)  # add space down
        plt.gcf().subplots_adjust(left=0.15)  # add space left
        plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
        ax = plt.gca()
        plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'])
        ax.tick_params(axis='x', which='major', width=7, length=24)
        ax.tick_params(axis='y', which='major', width=7, length=24, pad=20)
        ax.set_xlim([0, 25])
        # plt.xlim(0, 25)
        # plt.show()
        plt.savefig('Plot_ECDF'+'/'+ metric +'_igs.pdf', bbox_inches='tight')
        plt.close()







