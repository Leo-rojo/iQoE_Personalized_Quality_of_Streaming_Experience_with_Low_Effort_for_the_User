import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

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

n_queries=60
regr_choosen='XGboost'


for metric in ['mae','rmse']:
    save_modality=[]
    save_modality_stdv=[]
    for ss in range(3):
        save_ts=[]
        save_ts_std=[]
        for ts in ['ts_bb','ts_th','ts_mpc']:
            save_ave_across_users_for_shuffle=[]
            for shuffle in [13,34,42,70,104]:
                save_ave_across_users=[]
                for u in range(32):
                    for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                        main_path=regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'
                        user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(
                            u)+ '/' + ts + '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')[ss]
                        save_ave_across_users.append(user_data.tolist())
                save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
            save_ts.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
            save_ts_std.append(np.std(save_ave_across_users_for_shuffle,axis=0))
        save_modality.append(save_ts) #savemodality = [[against him],against_one]-->[ts_bb,ts_th..]
        save_modality_stdv.append(save_ts_std)

    main_path_for_save_fig = 'Plot_allusers_' + regr_choosen
    if not os.path.exists(main_path_for_save_fig):
        os.makedirs(main_path_for_save_fig)
    #np.save(main_path_for_save_fig + '/' + metric + 'values', save_modality)
    #np.save(main_path_for_save_fig + '/' + metric + 'values' + flag_insca, save_five_ss_stdv)
    #savemodality[againsthim,one,two][th_bb]
    for i in range(3):#ts_bb,ts_th,ts_mpc
        tr_abr=['bb', 'th', 'mpc']
        fig = plt.figure(figsize=(20, 10), dpi=100)
        leggs = ['bb', 'th', 'mpc']
        leggs.remove(tr_abr[i])
        leg = leggs
        colors = ['r', 'g', 'b']
        conta=0
        for againstwho in range(3): #[against him,against 1,against 2]
            users_meanmean = save_modality[conta][i]
            data1 = plt.scatter(range(n_queries + 1), users_meanmean, marker='.', color=colors[conta])
            # plt.errorbar(range(n_queries+1), users_meanmean, yerr=users_meanster, ls='none', color='k')#
            f = interp1d(range(n_queries + 1), users_meanmean)
            if conta==0:
                plt.plot(range(n_queries + 1), f(range(n_queries + 1)), '-', linewidth='2', label='train_'+tr_abr[i]+'_'+'test_'+tr_abr[i],
                         color=colors[conta])
            if conta==1:
                plt.plot(range(n_queries + 1), f(range(n_queries + 1)), '-', linewidth='2', label='train_'+tr_abr[i]+'_'+'test_'+leg[0],
                             color=colors[conta])
            if conta==2:
                plt.plot(range(n_queries + 1), f(range(n_queries + 1)), '-', linewidth='2', label='train_'+tr_abr[i]+'_'+'test_'+leg[1],
                             color=colors[conta])
            conta+=1
        # plt.grid()
        plt.xlabel("# of queries", fontdict=font_axes_titles)
        plt.xticks([0, 50, 100, 150, 200, 250], ['1', '50', '100', '150', '200', '250+'])
        plt.ylabel(metric.upper(), fontdict=font_axes_titles)
        #if metric == 'mae':
        #    plt.yticks(range(0, 6))
        plt.gcf().subplots_adjust(bottom=0.2)  # add space down
        plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
        # plt.show()
        # plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
        # plt.legend(ncol=5,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
        plt.legend(fontsize=40)
        # plt.show()
        plt.savefig(main_path_for_save_fig + '/' + metric + ['_ts_bb','_ts_th','_ts_mpc'][i]+ '.pdf', bbox_inches='tight')
        plt.close()


        ####barplot
        withhim=[]
        for i in save_modality[0]:
            withhim.append(i[49])
        withone = []
        for i in save_modality[1]:
            withone.append(i[49])
        withtwo = []
        for i in save_modality[2]:
            withtwo.append(i[49])

        one=withhim[0],withone[1],withone[2]
        two=withone[0],withhim[1],withtwo[2]
        three=withtwo[0],withtwo[1],withhim[2]
        np.save(main_path_for_save_fig + '/' + metric + 'values', [one,two,three])
        #std
        withhimstd = []
        for i in save_modality_stdv[0]:
            withhimstd.append(i[49])
        withonestd = []
        for i in save_modality_stdv[1]:
            withonestd.append(i[49])
        withtwostd = []
        for i in save_modality_stdv[2]:
            withtwostd.append(i[49])

        onestd = withhimstd[0], withonestd[1], withonestd[2]
        twostd = withonestd[0], withhimstd[1], withtwostd[2]
        threestd = withtwostd[0], withtwostd[1], withhimstd[2]

        fig = plt.figure(figsize=(20, 10), dpi=100)
        barWidth = 0.4
        a = [1,3,5]#np.arange(0, 3, 1)
        b = [i + barWidth+0.03 for i in a]
        c = [i + barWidth+0.03 for i in b]
        plt.bar(a, one, color='r', width=barWidth, linewidth=0, label='tested on BB',align='edge',yerr=onestd)
        plt.bar(b, two, color='g', width=barWidth, linewidth=0, label='tested on th', align='edge',yerr=twostd)
        plt.bar(c, three, color='b', width=barWidth, linewidth=0, label='tested on MPC', align='edge',yerr=threestd)
        plt.xticks([x+0.2 for x in b], ['BBA', 'TR', 'MPC'])
        plt.xlabel("Training ABR", fontdict=font_axes_titles)
        # plt.xlabel("Synthetic models",fontdict=font_axes_titles)
        plt.ylabel(metric.upper(), fontdict=font_axes_titles)
        plt.yticks(range(0, 12, 2))
        plt.gcf().subplots_adjust(bottom=0.2)  # add space down
        plt.gcf().subplots_adjust(left=0.15)  # add space left
        plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
        #plt.axhline(y=0, color='black', linestyle='-')
        ax = plt.gca()
        ax.tick_params(axis='x', which='major', width=7, length=24)
        ax.tick_params(axis='y', which='major', width=7, length=24)
        ax.set_ylim([0, 10])
        # plt.title('Comparison nr chunks '+str(nr_chunk),fontdict=font_title)
        #plt.legend(ncol=3)
        plt.show()
        plt.savefig(main_path_for_save_fig + '/' +'barplotabr' + metric + '.pdf', bbox_inches='tight')

        plt.close()