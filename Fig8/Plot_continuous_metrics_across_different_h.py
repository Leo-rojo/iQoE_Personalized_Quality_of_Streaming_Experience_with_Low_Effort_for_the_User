import numpy as np
import matplotlib.pyplot as plt
import os
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
    for ts in [1,10,20,30,40,50]:#,60,70,80]:#,50,60,70,80]:
        save_ave_across_users_for_shuffle=[]
        for shuffle in [13,34,42,70,104]:
            save_ave_across_users=[]
            for u in range(32):
                for QoE_model in ['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']:
                    main_path=regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_7_'+str(ts)
                    user_data = np.load(main_path + '/' + QoE_model + '/user_' + str(
                        u) +'/ts_'+str(ts)+'/' +'shuffle_'+str(shuffle) +'/'+ metric + '/scores_for_ALstrat.npy')#[ss]  # [0] is ave of shuffle
                    save_ave_across_users.append(user_data.tolist())
            save_ave_across_users_for_shuffle.append(np.mean(save_ave_across_users,axis=0))
        save_five_ss.append(np.mean(save_ave_across_users_for_shuffle,axis=0))
        save_five_ss_stdv.append(np.std(save_ave_across_users_for_shuffle,axis=0))

    main_path_for_save_fig = 'Plot_allusers_' + regr_choosen
    if not os.path.exists(main_path_for_save_fig):
        os.makedirs(main_path_for_save_fig)
    np.save(main_path_for_save_fig+'/'+metric+'values',save_five_ss)
    #np.save(main_path_for_save_fig + '/' + metric+'values'+flag_insca, save_five_ss_stdv)

    fig = plt.figure(figsize=(20, 10),dpi=100)
    leg = ['1', '10', '20', '30','40', '50']#,'60']#, '50','60','70','80']
    #colors=['c','b','r','m','g','y','sandybrown','pink','lightgreen']
    colors=[colori[0],colori[2],'r',colori[1],colori[4],colori[5],colori[6]]
    conta = 0
    users_meanmean_50=[]
    users_meanmean_100=[]
    users_meanmean_150=[]
    stile = ['-', '--', '-.', ':']
    for initial_number in leg:

        users_meanmean_50.append(save_five_ss[conta][0][49])
        users_meanmean_100.append(save_five_ss[conta][0][74])
        users_meanmean_150.append(save_five_ss[conta][0][99])
        #users_meanster=save_five_ss_stdv[conta][0][49]
        #data1 = plt.scatter(range(n_queries+1), users_meanmean, marker='.',color=colors[conta] )
        #plt.errorbar(range(n_queries+1), users_meanmean, yerr=users_meanster, ls='none', color='k')#
        #f = interp1d(range(n_queries+1), users_meanmean)
        conta += 1
    plt.plot(users_meanmean_50, '-', linewidth='7',label='after 50 SAs', color='r')#, label=leg[conta], color=colors[conta])'r',colori[2],colori[4]
    plt.plot(users_meanmean_100, '--', linewidth='7',label='after 100 SAs', color=colori[2],)  # , label=leg[conta], color=colors[conta])
    plt.plot(users_meanmean_150, ':', linewidth='7',label='after 150 SAs',color=colori[0])  # , label=leg[conta], color=colors[conta])
    #plt.grid()
    plt.xlabel("h, number of initial random samples", fontdict=font_axes_titles)
    plt.xticks(np.arange(6),['1','10','20','30','40','50'])#,'60','70','80'])
    plt.ylabel(metric.upper(), fontdict=font_axes_titles)

    plt.yticks(range(0,10,2))
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=7, length=24)
    ax.tick_params(axis='y', which='major', width=7, length=24)
    #yticks = ax.yaxis.get_major_ticks()
    #yticks[0].set_visible(False)
    #yticks[0].label1.set_visible(False)
    ax.set_ylim([2,7.2])
    #plt.show()
    #plt.title('Nc_' + str(nr_chunks) + ' QoEm_' + QoE_model, fontdict=font_title)
    #plt.legend(ncol=5,fontsize=20,loc='upper center',bbox_to_anchor=(0.48, 1.15),frameon=False)
    #handles = [Patch(facecolor=color, label=label)for label, color in zip(leg, colors)]
    #plt.legend(ncol=3, frameon=False,handlelength=2., handleheight=0.7, fontsize=40,handletextpad=0.1, columnspacing=0.5)
    #plt.legend(ncol=3, frameon=False, handles=handles, handlelength=2., handleheight=0.7, fontsize=40,bbox_to_anchor=(0.03, 0.07, 1, 1), handletextpad=0.1, columnspacing=0.5)
    #plt.show()
    plt.savefig(main_path_for_save_fig + '/' + metric+'_rs_lines.pdf',bbox_inches='tight')
    plt.close()



