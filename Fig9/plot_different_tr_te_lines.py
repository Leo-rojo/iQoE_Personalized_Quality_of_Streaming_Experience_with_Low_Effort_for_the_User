import numpy as np
import matplotlib.pyplot as plt
import os
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

n_queries=99
regr_choosen='XGboost'

for metric in ['mae','rmse']:
    save_ts=[]
    save_tsstdv=[]
    for t_s in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
        save_ave_across_users_for_shuffle=[]
        for shuffle in [13,34,42,70,104]:
            save_ave_across_users=[]
            for u in range(32):
                for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                    main_path=regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_1'
                    user_data = np.load('./' + main_path + '/' + QoE_model + '/user_' + str(
                        u) +'/ts_'+str(t_s)+ '/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')#[ss]  # [0] is ave of shuffle
                    save_ave_across_users.append(user_data.tolist())
            save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
        only100=np.mean(save_ave_across_users_for_shuffle, axis=0)
        only100std=np.std(save_ave_across_users_for_shuffle, axis=0)
        save_ts.append(only100[0])
        save_tsstdv.append(only100std[0])

    main_path_for_save_fig = 'tr_te' + regr_choosen
    if not os.path.exists(main_path_for_save_fig):
        os.makedirs(main_path_for_save_fig)
    #np.save(main_path_for_save_fig+'/'+metric+'values',save_ts)
    #np.save(main_path_for_save_fig + '/' + metric+'values'+flag_insca, save_tsstdv)

    # fig = plt.figure(figsize=(20, 10),dpi=100)
    leg = ['0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8','0.9']
    # #colors=['r','g','b','c']
    # conta = 0
    # for kind_strategy in leg:
    #     kind_strategy_idx=leg.index(kind_strategy)
    #     users_meanmean=save_ts[conta][0:100]
    #     users_meanmeanstd=save_tsstdv[conta][0:100]
    #     #users_meanster=save_five_ss_stdv[conta]
    #     data1 = plt.scatter(range(60+1), users_meanmean, marker='.')#,color=colors[conta] )
    #     #plt.errorbar(range(n_queries+1), users_meanmean, yerr=users_meanster, ls='none', color='k')#
    #     f = interp1d(range(60+1), users_meanmean)
    #     plt.plot(range(60+1), f(range(60+1)), '-', linewidth='2', label=leg[conta])#,color=colors[conta])
    #     conta += 1
    # plt.grid()
    # plt.xlabel("# of queries", fontdict=font_axes_titles)
    # plt.xticks([0, 50, 100],['1', '50', '100'])#, ['1', '50', '100', '150', '200', '250+'])
    # plt.ylabel(metric.upper(), fontdict=font_axes_titles)
    # plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    # plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    # #plt.show()
    # #plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
    # #plt.legend(ncol=5,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
    # plt.legend(fontsize=20)
    # #plt.show()
    # plt.savefig(main_path_for_save_fig + '/' + metric+'.png',bbox_inches='tight')
    # plt.close()

    #barre
    nr_query_plot=49
    conta=0
    users_meanmean_50=[]
    users_meanmean_75 = []
    users_meanmean_100 = []
    users_meanmeanstd=[]
    fig = plt.figure(figsize=(20, 10), dpi=100)
    for kind_strategy in leg:
        kind_strategy_idx=leg.index(kind_strategy)
        users_meanmean_50.append(save_ts[conta][49])
        users_meanmean_75.append(save_ts[conta][74])
        users_meanmean_100.append(save_ts[conta][99])
        #users_meanmeanstd.append(save_tsstdv[conta][nr_query_plot])
        conta += 1
    np.save(main_path_for_save_fig + '/' + metric + 'values_50', users_meanmean_50)
    np.save(main_path_for_save_fig + '/' + metric + 'values_75', users_meanmean_75)
    np.save(main_path_for_save_fig + '/' + metric + 'values_100', users_meanmean_100)
    plt.plot(users_meanmean_50, '-', linewidth='7', label='after 50 SAs',color='r')  # , label=leg[conta], color=colors[conta])'r',colori[2],colori[4]
    plt.plot(users_meanmean_75, '--', linewidth='7', label='after 100 SAs',color=colori[2], )  # , label=leg[conta], color=colors[conta])
    plt.plot(users_meanmean_100, ':', linewidth='7', label='after 150 SAs',color=colori[0])  # , label=leg[conta], color=colors[conta])
    #plt.grid()
    plt.xlabel("Training-set share", fontdict=font_axes_titles)
    plt.xticks(range(9), ['0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8','0.9'])  # , ['1', '50', '100', '150', '200', '250+'])
    plt.ylabel(metric.upper(), fontdict=font_axes_titles)
    plt.yticks(range(0, 10, 2))
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=7, length=24)
    ax.tick_params(axis='y', which='major', width=7, length=24)
    ax.set_ylim([2,8])
    # if metric=='mae':
    #     plt.yticks(np.arange(0,3.5,0.5))
    # plt.show()
    # plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
    # plt.legend(ncol=5,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
    #plt.legend(fontsize=20)
    # plt.show()
    plt.savefig(main_path_for_save_fig + '/' + metric +  '_line_'+'.pdf', bbox_inches='tight')
    plt.close()





