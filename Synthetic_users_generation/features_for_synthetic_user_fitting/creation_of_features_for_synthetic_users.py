import numpy as np

#collect all features and scores
all_features=np.load('../features_and_scores_WIV_hdtv_users/all_feat_hdtv.npy')

#calculate the minimum bitrate possible in WIV ladder
all_bitrates=[]
for exp in all_features:
    #bitrate
    for i in range(2,81,13):
        all_bitrates.append(float(exp[i]))
min_bit=np.array(all_bitrates).min()

#store features composition for each model for synthetic users
collect_sumbit=[]
collect_sumpsnr=[]
collect_sumssim=[]
collect_sumvmaf=[]

collect_logbit=[]
collect_FTW=[]
collect_SDNdash=[]
collect_videoAtlas=[]

for exp in all_features:

    # bitrate_features
    bit = []
    logbit=[]
    for i in range(2, 81, 13):
        bit.append(float(exp[i]))
        bit_log = np.log(float(exp[i]) / min_bit)
        logbit.append(bit_log)
    #sumbit
    s_bit = np.array(bit).sum()
    #sumlogbit
    l_bit = np.array(logbit).sum()

    #rebuffer
    reb=[]
    for i in range(1, 80, 13):
        reb.append(float(exp[i]))
    #sum of all reb
    s_reb=np.array(reb).sum()
    #ave of all reb
    s_reb_ave=np.array(reb).mean()
    #nr of stall
    nr_stall=np.count_nonzero(reb)
    #duration stall+normal
    tot_dur_plus_reb=7*4+s_reb  #nr chunks x duration chunk

    #psnr
    psnr = []
    for i in range(10, 89, 13):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()

    #ssim
    ssim = []
    for i in range(11, 90, 13):
        ssim.append(float(exp[i]))
    s_ssim = np.array(ssim).sum()

    #vmaf
    vmaf = []
    for i in range(12, 91, 13):
        vmaf.append(float(exp[i]))
    #sum
    s_vmaf = np.array(vmaf).sum()
    #ave
    s_vmaf_ave = np.array(vmaf).mean()

    #is best features for videoAtlas
    # isbest
    isbest = []
    for i in range(8, 87, 13):
        isbest.append(float(exp[i]))

    is_best=np.array(isbest)
    m = 0
    for idx in range(is_best.size - 1, -1, -1):
        if is_best[idx]:
            m += 4
        if reb[idx] > 0 or is_best[idx] == 0:
            break
    m /= tot_dur_plus_reb
    i = (np.array([4 for i in is_best if i == 0]).sum() + s_reb) / tot_dur_plus_reb

    #differnces
    s_dif_bit=np.abs(np.array(bit[1:]) - np.array(bit[:-1])).sum()
    s_dif_bitlog=np.abs(np.array(logbit[1:]) - np.array(logbit[:-1])).sum()
    s_dif_psnr=np.abs(np.array(psnr[1:]) - np.array(psnr[:-1])).sum()
    s_dif_ssim=np.abs(np.array(ssim[1:]) - np.array(ssim[:-1])).sum()
    s_dif_vmaf=np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).sum()
    a_dif_vmaf=np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).mean()

    #collection
    collect_sumbit.append([s_bit,s_reb,s_dif_bit])
    collect_sumpsnr.append([s_psnr, s_reb, s_dif_psnr])
    collect_sumssim.append([s_ssim, s_reb, s_dif_ssim])
    collect_sumvmaf.append([s_vmaf, s_reb, s_dif_vmaf])

    collect_logbit.append([l_bit, s_reb, s_dif_bitlog])
    collect_FTW.append([s_reb_ave,nr_stall])
    collect_SDNdash.append([s_vmaf_ave,s_reb_ave,a_dif_vmaf]) #without initial stall since we don't have it in our dataset
    collect_videoAtlas.append([s_vmaf_ave,s_reb/tot_dur_plus_reb,nr_stall,m,i])

#save data
np.save('feat_bit',collect_sumbit)
np.save('feat_psnr',collect_sumpsnr)
np.save('feat_ssim',collect_sumssim)
np.save('feat_vmaf',collect_sumvmaf)
np.save('feat_logbit',collect_logbit)
np.save('feat_FTW',collect_FTW)
np.save('feat_SDNdash',collect_SDNdash)
np.save('feat_videoAtlas',collect_videoAtlas)
