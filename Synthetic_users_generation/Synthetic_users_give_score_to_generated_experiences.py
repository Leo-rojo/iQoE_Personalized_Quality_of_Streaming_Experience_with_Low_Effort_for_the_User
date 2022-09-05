import pickle
import numpy as np
import random
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm # recommended import according to the docs
import os

#PARAMS
nr_of_exp=1000
nr_c=7


synthetic_experiences=np.load('./experiences_with_features.npy')
#modifiy bitrate form bit/s to kbit/s like in W4
for i in range(len(synthetic_experiences)):
    for k in range(len(synthetic_experiences[0])):
        synthetic_experiences[i][k][2]=synthetic_experiences[i][k][2]/1000

#inf to 50 for psnr--limit inf
synthetic_experiences[synthetic_experiences>1e308]=50
def sigmoid(x, k, x0):
    return (99.0 / (1 + np.exp(-k * (x - x0)))) + 1

#collect experiences in form of elaborated features
random.seed(42)

list_of_exp=[]
list_of_exp_for_models=[]
for i in range(nr_of_exp):
    random_trace=randint(0, len(synthetic_experiences)-1)
    random_chunk=randint(0, len(synthetic_experiences[0])-nr_c)
    #list of exp to be used for training
    ch=[]
    for c in range(nr_c):
        ch=ch + synthetic_experiences[random_trace][random_chunk+c].tolist()
    list_of_exp.append(ch)

np.save('./features_generated_experiences/'+'feat_iQoE_for_synth_exp',list_of_exp)

#calculate min bit
all_bitrates=[]
for exp in list_of_exp:
    #bitrate
    for i in range(2, (2+nr_c*10-1), 10):
        all_bitrates.append(float(exp[i]))
min_bit=np.array(all_bitrates).min()

#collect features my experience
collect_sumbit=[]
collect_sumpsnr=[]
collect_sumssim=[]
collect_sumvmaf=[]

collect_logbit=[]
collect_FTW=[]
collect_SDNdash=[]
collect_videoAtlas=[]
for exp in list_of_exp: #remember is long as two chunk all togheter

    # bitrate_features
    bit = []
    logbit = []
    for i in range(2, (2+nr_c*10-1), 10):
        bit.append(float(exp[i]))
        bit_log = np.log(float(exp[i]) / min_bit)
        logbit.append(bit_log)
    # sumbit
    s_bit = np.array(bit).sum()
    # sumlogbit
    l_bit = np.array(logbit).sum()

    # !!!!!!!!!!!!!!!rebuffer NB. [ch1],[ch2],[ch3] the stall in between ch1 e ch2 is contained in ch2 same for ch2 e ch3 that is contained in ch3
    reb = []
    for i in range(11, (1+nr_c*10-1), 10): #start from 11 and not 1 exactly for the reason explained before
        reb.append(float(exp[i]))
    # sum of all reb
    s_reb = np.array(reb).sum()
    # ave of all reb
    s_reb_ave = np.array(reb).mean()
    # nr of stall
    nr_stall = np.count_nonzero(reb)
    # duration stall+normal
    tot_dur_plus_reb = nr_c*4 + s_reb

    # psnr
    psnr = []
    for i in range(7,(7+nr_c*10-1), 10):
        psnr.append(float(exp[i]))
    s_psnr = np.array(psnr).sum()

    # ssim
    ssim = []
    for i in range(8, (8+nr_c*10-1), 10):
        ssim.append(float(exp[i]))
    s_ssim = np.array(ssim).sum()

    # vmaf
    vmaf = []
    for i in range(9, (9+nr_c*10-1), 10):
        vmaf.append(float(exp[i]))
    # sum
    s_vmaf = np.array(vmaf).sum()
    # ave
    s_vmaf_ave = np.array(vmaf).mean()

    # is best features for videoAtlas
    # isbest
    isbest = []
    for i in range(6, (6+nr_c*10-1), 10):
        isbest.append(float(exp[i]))

    is_best = np.array(isbest)
    m = 0
    for idx in range(is_best.size - 1, -1, -1):
        if is_best[idx]:
            m += 4
        rebatl=[0]+reb
        if rebatl[idx] > 0 or is_best[idx] == 0:
            break
    m /= tot_dur_plus_reb
    i = (np.array([4 for i in is_best if i == 0]).sum() + s_reb) / tot_dur_plus_reb

    # differnces
    s_dif_bit = np.abs(np.array(bit[1:]) - np.array(bit[:-1])).sum()
    s_dif_bitlog = np.abs(np.array(logbit[1:]) - np.array(logbit[:-1])).sum()
    s_dif_psnr = np.abs(np.array(psnr[1:]) - np.array(psnr[:-1])).sum()
    s_dif_ssim = np.abs(np.array(ssim[1:]) - np.array(ssim[:-1])).sum()
    s_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).sum()
    a_dif_vmaf = np.abs(np.array(vmaf[1:]) - np.array(vmaf[:-1])).mean()

    # collection
    collect_sumbit.append([s_bit, s_reb, s_dif_bit])
    collect_sumpsnr.append([s_psnr, s_reb, s_dif_psnr])
    collect_sumssim.append([s_ssim, s_reb, s_dif_ssim])
    collect_sumvmaf.append([s_vmaf, s_reb, s_dif_vmaf])

    collect_logbit.append([l_bit, s_reb, s_dif_bitlog])
    collect_FTW.append([s_reb_ave, nr_stall])
    collect_SDNdash.append([s_vmaf_ave, s_reb_ave, a_dif_vmaf])  # without initial stall since we don't have it in our dataset
    collect_videoAtlas.append([s_vmaf_ave, s_reb / tot_dur_plus_reb, nr_stall, m, i])

np.save('feat_bit_for_synth_exp',collect_sumbit)
np.save('feat_psnr_for_synth_exp',collect_sumpsnr)
np.save('feat_ssim_for_synth_exp',collect_sumssim)
np.save('feat_vmaf_for_synth_exp',collect_sumvmaf)
np.save('feat_logbit_for_synth_exp',collect_logbit)
np.save('feat_ftw_for_synth_exp',collect_FTW)
np.save('feat_sdn_for_synth_exp',collect_SDNdash)
np.save('feat_va_for_synth_exp',collect_videoAtlas)

#give scores
models=['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']
models_folder='Fitted_models_without_logistic'
params_sigmoid=np.load('./save_param_sigmoids/params_sigmoid.npy')

#load models (which means parameters)
all_synthetic_users=[]
for u in range(32):
    synthetic_user_models=[]
    for model in models:
        if model=='videoAtlas':
            with open('./'+models_folder+'/organized_by_users/user_'+str(u)+'/model_videoAtlas.pkl', 'rb') as handle:
                synthetic_user_models.append(pickle.load(handle))
        else:
            synthetic_user_models.append(np.load('./'+models_folder+'/organized_by_users/user_'+str(u)+'/model_'+model+'.npy',allow_pickle=True))
    all_synthetic_users.append(synthetic_user_models)
#['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']

##extract subexperiences from generated experiences
#order features: representation_index, rebuffering_duration, video_bitrate,
#                chunk_size, width,	height, is_best, psnr, ssim, vmaf
scores_by_models=[]
scores_by_models_not_scaled=[]
for u in range(32):
    scores_by_users = []
    scores_by_users_not_scaled = []
    user_models=all_synthetic_users[u]
    for kind_of_models in models:
        if kind_of_models=='bit':#[s_bit,s_dif_bit,s_psnr,s_dif_psnr,s_ssim,s_dif_ssim,s_vmaf,s_dif_vmaf,s_bit_log,s_dif_bit_log,ave_st_FTW,nr_stall_FTW,s_reb]
            temp_score=[]
            for exp in range(nr_of_exp):
                score=np.dot(user_models[0],collect_sumbit[exp]) #here should go the non linear mapping eventually for real context.
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            b_sig, c_sig = params_sigmoid[u][0]
            scores_by_users.append(sigmoid(temp_score,b_sig,c_sig))
        elif kind_of_models=='logbit':
            temp_score = []
            for exp in range(nr_of_exp):
                score=np.dot(user_models[1], collect_logbit[exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            b_sig, c_sig = params_sigmoid[u][1]
            scores_by_users.append(sigmoid(temp_score, b_sig, c_sig))
        elif kind_of_models=='psnr':
            temp_score = []
            for exp in range(nr_of_exp):
                score=np.dot(user_models[2], collect_sumpsnr[exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            b_sig, c_sig = params_sigmoid[u][2]
            scores_by_users.append(sigmoid(temp_score, b_sig, c_sig))
        elif kind_of_models=='ssim':
            temp_score = []
            for exp in range(nr_of_exp):
                score = np.dot(user_models[3], collect_sumssim[exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            b_sig, c_sig = params_sigmoid[u][3]
            scores_by_users.append(sigmoid(temp_score, b_sig, c_sig))
        elif kind_of_models=='vmaf':
            temp_score = []
            for exp in range(nr_of_exp):
                score = np.dot(user_models[4], collect_sumvmaf[exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            b_sig, c_sig = params_sigmoid[u][4]
            scores_by_users.append(sigmoid(temp_score, b_sig, c_sig))
        elif kind_of_models=='FTW':
            temp_score = []
            for exp in range(nr_of_exp):
                a, b, c, d = user_models[5]
                x1, x2 = collect_FTW[exp]
                score = a * np.exp(-(b * x1 + c) * x2) + d
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            #scores_by_users.append((np.array(temp_score))) #FTW not scaled, because already scaled
            b_sig, c_sig = params_sigmoid[u][5]
            scores_by_users.append(sigmoid(temp_score, b_sig, c_sig))
        elif kind_of_models=='SDNdash':
            temp_score = []
            for exp in range(nr_of_exp):
                score = np.dot(user_models[6], collect_SDNdash[exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            b_sig, c_sig = params_sigmoid[u][6]
            scores_by_users.append(sigmoid(temp_score, b_sig, c_sig))
        elif kind_of_models== 'videoAtlas':
            temp_score=[]
            with open('./' + models_folder + '/organized_by_users/user_' + str(u) + '/model_videoAtlas.pkl','rb') as handle:
                pickled_atlas=pickle.load(handle)
            videoAtlasregressor=pickled_atlas #0 there is the mdoel,
            temp_score = videoAtlasregressor.predict(collect_videoAtlas)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1,1)))
            b_sig, c_sig = params_sigmoid[u][7]
            scores_by_users.append(sigmoid(temp_score, b_sig, c_sig))

    #FTW metti in scaled anche se non scaled
    scores_by_models.append(scores_by_users)
    scores_by_models_not_scaled.append(scores_by_users_not_scaled)

np.save('./synthetic_users_scores_for_generated_experiences/scaled/nrchunks_'+str(nr_c),scores_by_models)
np.save('./synthetic_users_scores_for_generated_experiences/not_scaled/nrchunks_'+str(nr_c),scores_by_models_not_scaled)

#plot
font_axes_titles = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'bold',
        'size': 25,
        }
font_title = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'bold',
        'size': 40,
        }
font_general = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 30}
plt.rc('font', **font_general)

#plot scaled
for m in range(8):
    fig=plt.figure('scores by model'+models[m],figsize=(19.2,10.8), dpi=100)
    for u in range(32):
        my_scores=scores_by_models[u][m]
    #plt.plot(range(1,451),user_score,'o')
    #plt.plot(user_score)
        ecdf = sm.distributions.ECDF(my_scores.reshape(1000))
        plt.step(ecdf.x, ecdf.y,label='user_'+str(u))
    plt.xlabel('score',fontdict=font_axes_titles)#fontsize=18)
    plt.ylabel('ECDF',fontdict=font_axes_titles)#fontsize=18)
    plt.title('synthetic users scores distribution '+models[m],fontdict=font_title)
    plt.legend(loc='upper left', ncol=4, fontsize=10)
    plt.savefig('./plots/yes_scaled_out/'+models[m]+'_nr_chunk'+str(nr_c),bbox_inches='tight')
    plt.close(fig)

#plt not scaled
for m in range(8):
    fig=plt.figure('scores by model'+models[m],figsize=(19.2,10.8), dpi=100)
    for u in range(32):
        my_scores=scores_by_models_not_scaled[u][m]
    #plt.plot(range(1,451),user_score,'o')
    #plt.plot(user_score)
        ecdf = sm.distributions.ECDF(my_scores.reshape(1000))
        plt.step(ecdf.x, ecdf.y,label='user_'+str(u))
    plt.xlabel('score',fontdict=font_axes_titles)#fontsize=18)
    plt.ylabel('ECDF',fontdict=font_axes_titles)#fontsize=18)
    plt.title('synthetic users scores distribution '+models[m],fontdict=font_title)
    plt.legend(loc='upper left', ncol=4, fontsize=10)
    plt.savefig('./plots/no_scaled_out/'+models[m]+'_nr_chunk'+str(nr_c),bbox_inches='tight')
    plt.close(fig)