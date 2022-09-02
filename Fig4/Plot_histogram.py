import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib import cm
colori=cm.get_cmap('tab10').colors


font_axes_titles = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 25,
                        }
font_title = {'family': 'sans-serif',
                        'color':  'black',
                        #'weight': 'bold',
                        'size': 25,
                        }
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 25}
plt.rc('font', **font_general)

#collect metrics for each models
sc=['ftw','psnr','ssim','va','vmaf','sdn','iQoE_g','iQoE_p','bit','logbit']
aves_mae=[]
stds_mae=[]
aves_rmse=[]
stds_rmse=[]
for score in sc:
    aves_mae.append(np.load('./mae_rmse_test_experiences/'+score+'_scores/mae_ave.npy'))
    stds_mae.append(np.load('./mae_rmse_test_experiences/' + score + '_scores/mae_std.npy'))
    aves_rmse.append(np.load('./mae_rmse_test_experiences/' + score + '_scores/rmse_ave.npy'))
    stds_rmse.append(np.load('./mae_rmse_test_experiences/' + score + '_scores/rmse_std.npy'))


###################names
qoe_model=['B','L','P','S','V','F','N','A']
qmr=np.repeat(qoe_model,32)
users_names=[]
for i in range(8):
    for k in range(32):
        users_names.append(qmr[32*i+k]+str(k+1))

###plot mae
#take 26 worst users for iQoE-p
idx_most_difficult=sorted(range(len(aves_mae[7])), key=lambda x: aves_mae[7][x])[-26:]
worst_users_names_=(np.array(users_names)[idx_most_difficult]).tolist()
idx_most_difficult=[58,90,129,194,170,1,81,249]
worst_users_names_=(np.array(users_names)[idx_most_difficult]).tolist()
worst_users_names = [worst_users_names_[i] for i in range(8)]  #[worst_users_names_[i] if i in [ 0,  3,  6,  9, 12, 15, 18, 21, 25] else "" for i in range(26)]
worst26_mosbit=[aves_mae[8][i] for i in idx_most_difficult]
worst26_mosbit_std=[stds_mae[8][i] for i in idx_most_difficult]
worst26_moslogbit=[aves_mae[9][i] for i in idx_most_difficult]
worst26_moslogbit_std=[stds_mae[9][i] for i in idx_most_difficult]
worst26_mosva=[aves_mae[3][i] for i in idx_most_difficult]
worst26_mosva_std=[stds_mae[3][i] for i in idx_most_difficult]
worst26_iqoeg=[aves_mae[6][i] for i in idx_most_difficult]
worst26_iqoeg_std=[stds_mae[6][i] for i in idx_most_difficult]
worst26_iqoep=[aves_mae[7][i] for i in idx_most_difficult]
worst26_iqoep_std=[stds_mae[7][i] for i in idx_most_difficult]
worst26_mosftw=[aves_mae[0][i] for i in idx_most_difficult]
worst26_mosftw_std=[stds_mae[0][i] for i in idx_most_difficult]
worst26_mossdn=[aves_mae[5][i] for i in idx_most_difficult]
worst26_mossdn_std=[stds_mae[5][i] for i in idx_most_difficult]
worst26_mosssim=[aves_mae[2][i] for i in idx_most_difficult]
worst26_mosssim_std=[stds_mae[2][i] for i in idx_most_difficult]
worst26_mospsnr=[aves_mae[1][i] for i in idx_most_difficult]
worst26_mospsnr_std=[stds_mae[1][i] for i in idx_most_difficult]
worst26_mosvmaf=[aves_mae[4][i] for i in idx_most_difficult]
worst26_mosvmaf_std=[stds_mae[4][i] for i in idx_most_difficult]
#save results
np.save('./store_results/mosva',worst26_mosva)
np.save('./store_results/iqoeg',worst26_iqoeg)
np.save('./store_results/iqoep',worst26_iqoep)
np.save('./store_results/mossdn',worst26_mossdn)
np.save('./store_results/mospsnr',worst26_mospsnr)
np.save('./store_results/mosssim',worst26_mosssim)
np.save('./store_results/mosvmaf',worst26_mosvmaf)
np.save('./store_results/mosftw',worst26_mosftw)

#plots mae histogram
for metric in ['mae']:
    fig = plt.figure(figsize=(20, 5),dpi=100)
    #plt.axhline(y=250, color='black', linestyle='-')
    barWidth = 1.75
    a = np.arange(0, 8*20, 20)
    b = [i + barWidth for i in a]
    c = [i + barWidth for i in b]
    d = [i + barWidth for i in c]
    e = [i + barWidth for i in d]
    f = [i + barWidth for i in e]
    g = [i + barWidth for i in f]
    h = [i + barWidth for i in g]
    l = [i + barWidth for i in h]
    m = [i + barWidth for i in l]
    #plt.bar(a, worst26_mosbit, color='c', width=barWidth - 0.1, linewidth=0, label='Bit-group',align='edge')  # yerr=ss_elab_std[2]
    plt.bar(a, worst26_iqoep, color='r', width=barWidth, linewidth=0, label='iQoE-personal_50q', align='edge',yerr=worst26_iqoep_std)
    plt.bar(b, worst26_mosbit, color=colori[1], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',yerr=worst26_mosbit_std)
    plt.bar(c, worst26_moslogbit, color=colori[2], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',yerr=worst26_moslogbit_std)
    plt.bar(d, worst26_mosvmaf, color=colori[8], width=barWidth, linewidth=0, label='VMAF-group', align='edge',yerr=worst26_mosvmaf_std)
    plt.bar(e, worst26_mospsnr, color=colori[7], width=barWidth, linewidth=0, label='PSNR-group', align='edge',yerr=worst26_mospsnr_std)
    plt.bar(f, worst26_mosssim, color=colori[6], width=barWidth, linewidth=0, label='SSIM-group', align='edge',yerr=worst26_mosssim_std)
    plt.bar(g, worst26_mosftw, color=colori[9], width=barWidth, linewidth=0, label='FTW-group',align='edge',yerr=worst26_mosftw_std)
    plt.bar(h, worst26_mosva, color=colori[4], width=barWidth, linewidth=0, label='VA-group', align='edge',yerr=worst26_mosva_std)
    plt.bar(l, worst26_mossdn, color='gold', width=barWidth, linewidth=0, label='SDN-group', align='edge',yerr=worst26_mossdn_std)
    plt.bar(m, worst26_iqoeg, color=colori[0], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',yerr=worst26_iqoeg_std)
    #plt.xticks(np.arange(8) + barWidth*2.1, ['Bit','Log','Ps','Ss','Vm','FTW','SDN','VA'])
    plt.xlabel("Users", fontdict=font_axes_titles)
    plt.ylabel(metric.upper(), fontdict=font_axes_titles)
    plt.yticks(range(0, 60, 10))
    worst_26_names=[]
    plt.xticks(np.arange(0, 8*20, 20) + barWidth*5, worst_users_names,fontsize=25)
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.gcf().subplots_adjust(left=0.15)
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=3, length=10)
    ax.tick_params(axis='y', which='major', width=3, length=10)
    ax.set_ylim([0, 45])
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    #plt.title('treshold metric '+metric +' '+regr_choosen , fontdict=font_title)
    #plt.legend(fontsize=23,frameon=False,loc='upper left')
    plt.savefig('./histogram_'+metric+'.pdf',bbox_inches='tight')
    plt.close()

###plot rmse
#take 26 worst users for iQoE-p
worst26_moslogbit_std_rmse=[stds_rmse[9][i] for i in idx_most_difficult]
worst26_mosbit_rmse=[aves_rmse[8][i] for i in idx_most_difficult]
worst26_mosbit_std_rmse=[stds_rmse[8][i] for i in idx_most_difficult]
worst26_moslogbit_rmse=[aves_rmse[9][i] for i in idx_most_difficult]
worst26_mosva_rmse=[aves_rmse[3][i] for i in idx_most_difficult]
worst26_mosva_std_rmse=[stds_rmse[3][i] for i in idx_most_difficult]
worst26_iqoeg_rmse=[aves_rmse[6][i] for i in idx_most_difficult]
worst26_iqoeg_std_rmse=[stds_rmse[6][i] for i in idx_most_difficult]
worst26_iqoep_rmse=[aves_rmse[7][i] for i in idx_most_difficult]
worst26_iqoep_std_rmse=[stds_rmse[7][i] for i in idx_most_difficult]
worst26_mosftw_rmse=[aves_rmse[0][i] for i in idx_most_difficult]
worst26_mosftw_std_rmse=[stds_rmse[0][i] for i in idx_most_difficult]
worst26_mossdn_rmse=[aves_rmse[5][i] for i in idx_most_difficult]
worst26_mossdn_std_rmse=[stds_rmse[5][i] for i in idx_most_difficult]
worst26_mosssim_rmse=[aves_rmse[2][i] for i in idx_most_difficult]
worst26_mosssim_std_rmse=[stds_rmse[2][i] for i in idx_most_difficult]
worst26_mospsnr_rmse=[aves_rmse[1][i] for i in idx_most_difficult]
worst26_mospsnr_std_rmse=[stds_rmse[1][i] for i in idx_most_difficult]
worst26_mosvmaf_rmse=[aves_rmse[4][i] for i in idx_most_difficult]
worst26_mosvmaf_std_rmse=[stds_rmse[4][i] for i in idx_most_difficult]
np.save('./store_results/mosva_rmse',worst26_mosva_rmse)
np.save('./store_results/iqoeg_rmse',worst26_iqoeg_rmse)
np.save('./store_results/iqoep_rmse',worst26_iqoep_rmse)
np.save('./store_results/mossdn_rmse',worst26_mossdn_rmse)
np.save('./store_results/mospsnr_rmse',worst26_mospsnr_rmse)
np.save('./store_results/mosssim_rmse',worst26_mosssim_rmse)
np.save('./store_results/mosvmaf_rmse',worst26_mosvmaf_rmse)
np.save('./store_results/mosftw_rmse',worst26_mosftw_rmse)
for metric in ['rmse']:
    fig = plt.figure(figsize=(20, 5), dpi=100)
    # plt.axhline(y=250, color='black', linestyle='-')
    barWidth = 1.75
    a = np.arange(0, 8 * 20, 20)
    b = [i + barWidth for i in a]
    c = [i + barWidth for i in b]
    d = [i + barWidth for i in c]
    e = [i + barWidth for i in d]
    f = [i + barWidth for i in e]
    g = [i + barWidth for i in f]
    h = [i + barWidth for i in g]
    l = [i + barWidth for i in h]
    m = [i + barWidth for i in l]#qoe_model=['B','L','V','P','S',,'F','A','N']
    # plt.bar(a, worst26_mosbit, color='c', width=barWidth - 0.1, linewidth=0, label='Bit-group',align='edge')  # yerr=ss_elab_std[2]
    plt.bar(a, worst26_iqoep_rmse, color='r', width=barWidth, linewidth=0, label='iQoE-personal_50q', align='edge',yerr=worst26_iqoep_std_rmse)
    plt.bar(b, worst26_mosbit_rmse, color=colori[1], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',
            yerr=worst26_mosbit_std_rmse)
    plt.bar(c, worst26_moslogbit_rmse, color=colori[2], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',
            yerr=worst26_moslogbit_std_rmse)
    plt.bar(d, worst26_mosvmaf_rmse, color=colori[8], width=barWidth, linewidth=0, label='VMAF-group', align='edge',
            yerr=worst26_mosvmaf_std_rmse)
    plt.bar(e, worst26_mospsnr_rmse, color=colori[7], width=barWidth, linewidth=0, label='PSNR-group', align='edge',
            yerr=worst26_mospsnr_std_rmse)
    plt.bar(f, worst26_mosssim_rmse, color=colori[6], width=barWidth, linewidth=0, label='SSIM-group', align='edge',
            yerr=worst26_mosssim_std_rmse)
    plt.bar(g, worst26_mosftw_rmse, color=colori[9], width=barWidth, linewidth=0, label='FTW-group', align='edge',
            yerr=worst26_mosftw_std_rmse)
    plt.bar(h, worst26_mosva_rmse, color=colori[4], width=barWidth, linewidth=0, label='VA-group', align='edge',
            yerr=worst26_mosva_std_rmse)
    plt.bar(l, worst26_mossdn_rmse, color='gold', width=barWidth, linewidth=0, label='SDN-group', align='edge',
            yerr=worst26_mossdn_std_rmse)
    plt.bar(m, worst26_iqoeg_rmse, color=colori[0], width=barWidth, linewidth=0, label='iQoE-group_50q', align='edge',
            yerr=worst26_iqoeg_std_rmse)

    # plt.xticks(np.arange(8) + barWidth*2.1, ['Bit','Log','Ps','Ss','Vm','FTW','SDN','VA'])
    plt.xlabel("Users", fontdict=font_axes_titles)
    plt.ylabel(metric.upper(), fontdict=font_axes_titles)
    plt.yticks(range(0, 60, 10))
    worst_26_names = []
    plt.xticks(np.arange(0, 8*20, 20) + barWidth*5, worst_users_names,fontsize=25)
    plt.gcf().subplots_adjust(bottom=0.2)  # add space down
    plt.gcf().subplots_adjust(left=0.15)
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', width=3, length=10)
    ax.tick_params(axis='y', which='major', width=3, length=10)
    ax.set_ylim([0, 45])
    plt.margins(0.02, 0.01)  # riduci margini tra plot e bordo
    # plt.title('treshold metric '+metric +' '+regr_choosen , fontdict=font_title)
    # plt.legend(fontsize=23,frameon=False,loc='upper left')
    plt.savefig('./histogram_' + metric + '.pdf', bbox_inches='tight')
    plt.close()