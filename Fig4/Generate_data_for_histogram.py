from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import xgboost as xgb
from modAL.models import ActiveLearner
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn import linear_model
from scipy.optimize import curve_fit

#be in folder fig4


#take all individual user scores saved previously
collect_all=[]
users_scores=np.load('./synthetic_users_scores_for_generated_experiences/scaled/nrchunks_7.npy')
users_scores=users_scores.reshape(256,1000)

#take mos scores
mosarray=np.mean(users_scores,axis=0) # da splittare in 70-30

#function to fit video_atlas
def fit_supreg(all_features,mosscore):
    data = np.array(all_features)
    target = np.array(mosscore)

    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
                             param_grid={'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
                                         'gamma': np.logspace(-2, 2, 15)})
    regressor.fit(data, np.ravel(target))

    return regressor.best_estimator_
#function to fit linear models
def fit_linear(all_features,mosscore):
    # multi-linear model fitting
    X = all_features
    y = mosscore

    lm = linear_model.LinearRegression(fit_intercept=False)
    model = lm.fit(X, y)

    alpha = lm.coef_[0]
    beta = lm.coef_[1]
    gamma = lm.coef_[2]

    return [alpha, beta, gamma]
#function to fit non linear models
def fit_nonlinear(all_features,mosscore):
    def fun(data,a, b, c, d):
        x1, x2 = data
        y = a * np.exp(-(b * x1 + c) * x2) + d
        return y

    # Fit the curve
    popt, pcov = curve_fit(fun, all_features, mosscore, maxfev=1000000)
    estimated_a, estimated_b, estimated_c, estimated_d = popt
    return estimated_a, estimated_b, estimated_c, estimated_d
#function for iGS
def random_greedy_sampling_input_output(regressor, X, switch):
    if not switch:
        n_samples = len(X)
        query_idx = np.random.choice(range(n_samples))
    else:
        y = regressor.predict(X)
        dist_x_matrix = pairwise_distances(regressor.X_training, X)
        dist_y_matrix = pairwise_distances(
            regressor.y_training.reshape(-1, 1), y.reshape(-1, 1)
        )
        dist_to_training_set = np.amin(dist_x_matrix * dist_y_matrix, axis=0)
        query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]

#load saved IFs for each model and for each generated experience
features_folder='features_generated_experiences'
#load all bitrate features
all_features_bit = np.load('./'+features_folder+'/feat_bit_for_synth_exp.npy')
#load all bitrate features
all_features_psnr = np.load('./'+features_folder+'/feat_psnr_for_synth_exp.npy')
#load all bitrate features
all_features_ssim = np.load('./'+features_folder+'/feat_ssim_for_synth_exp.npy')
#load all bitrate features
all_features_vmaf = np.load('./'+features_folder+'/feat_vmaf_for_synth_exp.npy')
#load all bitrate features
all_features_sdn = np.load('./'+features_folder+'/feat_sdn_for_synth_exp.npy')
#load all bitrate features
all_features_logbit = np.load('./'+features_folder+'/feat_logbit_for_synth_exp.npy')
#load all bitrate features
all_features_ftw = np.load('./'+features_folder+'/feat_ftw_for_synth_exp.npy')
#load all videoatlas features
all_features_va = np.load('./'+features_folder+'/feat_videoAtlas_for_synth_exp.npy')
#load all features iQoE
all_features_iQoE = np.load('./'+features_folder+'/feat_iQoE_for_synth_exp.npy')

#run for 5 different seeds for reducing the impact of random shuffles
for rs in [42,13,70,34,104]:
    #split train test for group models, iQoE-group and iQoE-personal
    X_train_va, X_test_va, y_train_va, y_test_va = train_test_split(all_features_va, mosarray, test_size=0.3, random_state=rs)
    #scale inputs
    scaler = MinMaxScaler()
    scaler.fit(X_train_va)
    X_train_va = scaler.transform(X_train_va)
    X_test_va = scaler.transform(X_test_va)
    X_train_iQoE_g, X_test_iQoE_g, y_train_iQoE_g, y_test_iQoE_g = train_test_split(all_features_iQoE, mosarray, test_size=0.3, random_state=rs)
    scaler = MinMaxScaler()
    scaler.fit(X_train_iQoE_g)
    X_train_iQoE_g = scaler.transform(X_train_iQoE_g)
    X_test_iQoE_g = scaler.transform(X_test_iQoE_g)
    #bit
    X_train_bit_g, X_test_bit_g, y_train_bit_g, y_test_bit_g = train_test_split(all_features_bit, mosarray, test_size=0.3, random_state=rs)
    #bit
    X_train_psnr_g, X_test_psnr_g, y_train_psnr_g, y_test_psnr_g = train_test_split(all_features_psnr, mosarray, test_size=0.3, random_state=rs)
    #bit
    X_train_ssim_g, X_test_ssim_g, y_train_ssim_g, y_test_ssim_g = train_test_split(all_features_ssim, mosarray, test_size=0.3, random_state=rs)
    #bit
    X_train_vmaf_g, X_test_vmaf_g, y_train_vmaf_g, y_test_vmaf_g = train_test_split(all_features_vmaf, mosarray, test_size=0.3, random_state=rs)
    #bit
    X_train_sdn_g, X_test_sdn_g, y_train_sdn_g, y_test_sdn_g = train_test_split(all_features_sdn, mosarray, test_size=0.3, random_state=rs)
    #bit
    X_train_ftw_g, X_test_ftw_g, y_train_ftw_g, y_test_ftw_g = train_test_split(all_features_ftw, mosarray, test_size=0.3, random_state=rs)
    #bit
    X_train_logbit_g, X_test_logbit_g, y_train_logbit_g, y_test_logbit_g = train_test_split(all_features_logbit, mosarray, test_size=0.3, random_state=rs)
    tr_te_each_users=[]
    for u in range(256):
        X_train_iQoE_p, X_test_iQoE_p, y_train_iQoE_p, y_test_iQoE_p = train_test_split(all_features_iQoE, users_scores[u], test_size=0.3, random_state=rs)
        scaler = MinMaxScaler()
        scaler.fit(X_train_iQoE_p)
        X_train_p = scaler.transform(X_train_iQoE_p)
        X_test_p = scaler.transform(X_test_iQoE_p)
        tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])

    #training groups
    model_trained_mos_va=fit_supreg(X_train_va,y_train_va)
    model_trained_mos_bit=fit_linear(X_train_bit_g,y_train_bit_g)
    model_trained_mos_psnr=fit_linear(X_train_psnr_g,y_train_psnr_g)
    model_trained_mos_ssim=fit_linear(X_train_ssim_g,y_train_ssim_g)
    model_trained_mos_vmaf=fit_linear(X_train_vmaf_g,y_train_vmaf_g)
    model_trained_mos_sdn=fit_linear(X_train_sdn_g,y_train_sdn_g)
    model_trained_mos_logbit=fit_linear(X_train_logbit_g,y_train_logbit_g)
    model_trained_mos_ftw=fit_nonlinear((X_train_ftw_g[:,0],X_train_ftw_g[:,1]),y_train_ftw_g)

    #train iQoE_g (iQoE trained with MOS scores)
    nr_feat=70 #nr chunksXfeatures
    X_train=X_train_iQoE_g
    y_train=y_train_iQoE_g
    n_initial = 1
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_init_training, y_init_training = X_train[initial_idx], np.array(y_train,dtype=int)[initial_idx]
    
    # Isolate the non-training examples we'll be querying.
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)
    regr_5 = xgb.XGBRegressor(n_estimators = 100, max_depth = 60,nthread=1)
    regressor_iGS = ActiveLearner(
        estimator=regr_5,
        query_strategy=random_greedy_sampling_input_output,
        X_training=X_init_training.reshape(-1, nr_feat),
        y_training=y_init_training.reshape(-1, 1).flatten()
    )
    X_pool_iGS=X_pool.copy()
    y_pool_iGS=y_pool.copy()
    
    # active learning
    n_queries=60
    t_s=20
    count_queries=1
    switch_bol=False
    for idx in range(n_queries):
        if count_queries>t_s:
            switch_bol=True
        # iGS
        query_idx, query_instance = regressor_iGS.query(X_pool_iGS,switch=switch_bol)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        regressor_iGS.teach(np.array(X_pool_iGS[query_idx]).reshape(-1, nr_feat),
                           np.array(y_pool_iGS[query_idx]).reshape(-1, 1).flatten())
        X_pool_iGS, y_pool_iGS = np.delete(X_pool_iGS, query_idx, axis=0), np.delete(y_pool_iGS, query_idx)
        print('training_query: '+str(count_queries))
        count_queries+=1
        if idx in [49]:
            regressor_iGS.estimator.save_model('./iQoE_g_models/iQoE_g_'+ str(idx)+str(rs)+'.json')

    #train iQoE_p for each synthetic users 
    save_trained_model_each_users=[]
    for u in range(256):
        nr_feat = 70  # nr chunksXfeatures
        ###Active_leanring### tr_te_each_users_iqoe.append([X_train_u_iqoe,X_test_u_iqoe,y_train_u_iqoe,y_test_u_iqoe])
        X_train = tr_te_each_users[u][0]
        y_train = tr_te_each_users[u][2]
        n_initial = 1
        initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
        X_init_training, y_init_training = X_train[initial_idx], np.array(y_train, dtype=int)[initial_idx]

        # Isolate the non-training examples we'll be querying.
        X_pool = np.delete(X_train, initial_idx, axis=0)
        y_pool = np.delete(y_train, initial_idx, axis=0)

        regr_5 = xgb.XGBRegressor(n_estimators=100, max_depth=60, nthread=1)

        regressor_iGS = ActiveLearner(
            estimator=regr_5,
            query_strategy=random_greedy_sampling_input_output,
            X_training=X_init_training.reshape(-1, nr_feat),
            y_training=y_init_training.reshape(-1, 1).flatten()
        )
        X_pool_iGS = X_pool.copy()
        y_pool_iGS = y_pool.copy()

        # active learning
        n_queries = 60
        t_s = 20
        count_queries = 1
        switch_bol = False
        for idx in range(n_queries):
            if count_queries > t_s:
                switch_bol = True
            # iGS
            query_idx, query_instance = regressor_iGS.query(X_pool_iGS,switch=switch_bol)
            # print('gs_' + str(query_idx))
            query_idx = int(query_idx)  # 0because it is a list in this particular case
            regressor_iGS.teach(np.array(X_pool_iGS[query_idx]).reshape(-1, nr_feat),
                                 np.array(y_pool_iGS[query_idx]).reshape(-1, 1).flatten())
            X_pool_iGS, y_pool_iGS = np.delete(X_pool_iGS, query_idx, axis=0), np.delete(y_pool_iGS, query_idx)
            print('training_query: ' + str(count_queries))
            count_queries += 1
            if idx in [49]:
                regressor_iGS.estimator.save_model('./iQoE_p_models/iQoE_p_'+str(u)+'_'+str(idx)+str(rs)+'.json')

    ########### it takes some time################## 
    #load the saved trained models for iQoE_p and iQoE_g
    n_q=49
    model_trained_mos_iqoe=xgb.XGBRegressor()
    model_trained_mos_iqoe.load_model('./iQoE_g_models/iQoE_g_'+str(n_q)+str(rs)+'.json')
    model_trained_p_iqoe=[]
    for u in range(256):
        model_trained_px = xgb.XGBRegressor()
        model_trained_px.load_model('./iQoE_p_models/iQoE_p_'+str(u)+'_'+str(n_q)+str(rs)+'.json')
        model_trained_p_iqoe.append(model_trained_px)

    #va-group prediction on test
    mosmodel_us_scores_mae=[]
    mosmodel_us_scores_rmse=[]
    for u in range(256):#tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_va=model_trained_mos_va.predict(X_test_va)
        mosmodel_us_scores_mae.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_va))
        mosmodel_us_scores_rmse.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_va)))
    np.save('./mae_rmse_test_experiences/va_scores/mae'+str(rs),mosmodel_us_scores_mae)
    np.save('./mae_rmse_test_experiences/va_scores/rmse' + str(rs), mosmodel_us_scores_rmse)

    #psnr-group prediction on test
    mosmodel_us_scores_mae_psnr=[]
    mosmodel_us_scores_rmse_psnr=[]
    for u in range(256):#tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_psnr=[np.dot(model_trained_mos_psnr,X_test_psnr_g[i]) for i in range(len(X_test_psnr_g))]
        mosmodel_us_scores_mae_psnr.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_psnr))
        mosmodel_us_scores_rmse_psnr.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_psnr)))
    np.save('./mae_rmse_test_experiences/psnr_scores/mae' + str(rs), mosmodel_us_scores_mae_psnr)
    np.save('./mae_rmse_test_experiences/psnr_scores/rmse' + str(rs), mosmodel_us_scores_rmse_psnr)
    
    #ssim-group prediction on test
    mosmodel_us_scores_mae_ssim=[]
    mosmodel_us_scores_rmse_ssim=[]
    for u in range(256):#tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_ssim=[np.dot(model_trained_mos_ssim,X_test_ssim_g[i]) for i in range(len(X_test_ssim_g))]
        mosmodel_us_scores_mae_ssim.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_ssim))
        mosmodel_us_scores_rmse_ssim.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_ssim)))
    np.save('./mae_rmse_test_experiences/ssim_scores/mae' + str(rs), mosmodel_us_scores_mae_ssim)
    np.save('./mae_rmse_test_experiences/ssim_scores/rmse' + str(rs), mosmodel_us_scores_rmse_ssim)
    
    #vmaf-group prediction on test
    mosmodel_us_scores_mae_vmaf=[]
    mosmodel_us_scores_rmse_vmaf=[]
    for u in range(256):#tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_vmaf=[np.dot(model_trained_mos_vmaf,X_test_vmaf_g[i]) for i in range(len(X_test_vmaf_g))]
        mosmodel_us_scores_mae_vmaf.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_vmaf))
        mosmodel_us_scores_rmse_vmaf.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_vmaf)))
    np.save('./mae_rmse_test_experiences/vmaf_scores/mae' + str(rs), mosmodel_us_scores_mae_vmaf)
    np.save('./mae_rmse_test_experiences/vmaf_scores/rmse' + str(rs), mosmodel_us_scores_rmse_vmaf)
    
    # bit-group prediction on test
    mosmodel_us_scores_mae_bit = []
    mosmodel_us_scores_rmse_bit = []
    for u in range(256):  # tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_bit = [np.dot(model_trained_mos_bit, X_test_bit_g[i]) for i in range(len(X_test_bit_g))]
        mosmodel_us_scores_mae_bit.append(mean_absolute_error(tr_te_each_users[u][3], user_u_scores_bit))
        mosmodel_us_scores_rmse_bit.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_bit)))
    np.save('./mae_rmse_test_experiences/bit_scores/mae' + str(rs), mosmodel_us_scores_mae_bit)
    np.save('./mae_rmse_test_experiences/bit_scores/rmse' + str(rs), mosmodel_us_scores_rmse_bit)
    
    # logbit-group prediction on test
    mosmodel_us_scores_mae_logbit = []
    mosmodel_us_scores_rmse_logbit = []
    for u in range(256):  # tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_logbit = [np.dot(model_trained_mos_logbit, X_test_logbit_g[i]) for i in
                                range(len(X_test_logbit_g))]
        mosmodel_us_scores_mae_logbit.append(mean_absolute_error(tr_te_each_users[u][3], user_u_scores_logbit))
        mosmodel_us_scores_rmse_logbit.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_logbit)))
    np.save('./mae_rmse_test_experiences/logbit_scores/mae' + str(rs), mosmodel_us_scores_mae_logbit)
    np.save('./mae_rmse_test_experiences/logbit_scores/rmse' + str(rs), mosmodel_us_scores_rmse_logbit)

    # sdn-group prediction on test
    mosmodel_us_scores_mae_sdn=[]
    mosmodel_us_scores_rmse_sdn=[]
    for u in range(256):#tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_sdn=[np.dot(model_trained_mos_sdn,X_test_sdn_g[i]) for i in range(len(X_test_sdn_g))]
        mosmodel_us_scores_mae_sdn.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_sdn))
        mosmodel_us_scores_rmse_sdn.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_sdn)))
    np.save('./mae_rmse_test_experiences/sdn_scores/mae' + str(rs), mosmodel_us_scores_mae_sdn)
    np.save('./mae_rmse_test_experiences/sdn_scores/rmse' + str(rs), mosmodel_us_scores_rmse_sdn)
    
    #ftw-group prediction on test
    mosmodel_us_scores_mae_ftw=[]
    mosmodel_us_scores_rmse_ftw=[]
    for u in range(256):#tr_te_each_users.append([X_train_iQoE_p,X_test_iQoE_p,y_train_iQoE_p,y_test_iQoE_p])
        user_u_scores_ftw=[]
        for i in range(len(X_test_ftw_g)):
            a, b, c, d = model_trained_mos_ftw
            x1, x2 = X_test_ftw_g[i]
            score = a * np.exp(-(b * x1 + c) * x2) + d
            user_u_scores_ftw.append(score)
        mosmodel_us_scores_mae_ftw.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_ftw))
        mosmodel_us_scores_rmse_ftw.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_ftw)))
    np.save('./mae_rmse_test_experiences/ftw_scores/mae' + str(rs), mosmodel_us_scores_mae_ftw)
    np.save('./mae_rmse_test_experiences/ftw_scores/rmse' + str(rs), mosmodel_us_scores_rmse_ftw)
    
    #iQoE-group prediction on test
    iqoemosmodel_us_scores_mae=[]
    iqoemosmodel_us_scores_rmse=[]
    for u in range(256):
        user_u_scores_iqoeg=model_trained_mos_iqoe.predict(X_test_iQoE_g)
        iqoemosmodel_us_scores_mae.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_iqoeg))
        iqoemosmodel_us_scores_rmse.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_iqoeg)))
    np.save('./mae_rmse_test_experiences/iQoE_g_scores/mae' + str(rs), iqoemosmodel_us_scores_mae)
    np.save('./mae_rmse_test_experiences/iQoE_g_scores/rmse' + str(rs), iqoemosmodel_us_scores_rmse)
    
    #iQoE-personal prediction on test
    iqoepmodel_us_scores_mae=[]
    iqoepmodel_us_scores_rmse=[]
    for u in range(256):
        user_u_scores_iqoep=model_trained_p_iqoe[u].predict(tr_te_each_users[u][1])#X_test_u
        iqoepmodel_us_scores_mae.append(mean_absolute_error(tr_te_each_users[u][3],user_u_scores_iqoep))
        iqoepmodel_us_scores_rmse.append(sqrt(mean_squared_error(tr_te_each_users[u][3], user_u_scores_iqoep)))
    np.save('./mae_rmse_test_experiences/iQoE_p_scores/mae' + str(rs), iqoepmodel_us_scores_mae)
    np.save('./mae_rmse_test_experiences/iQoE_p_scores/rmse' + str(rs), iqoepmodel_us_scores_rmse)

#aggregate results for the 5 different shuffles
sc=['bit','logbit','ftw','psnr','ssim','va','vmaf','sdn','iQoE_g','iQoE_p']
for metric in ['mae','rmse']:
    for score in sc:
        each=[]
        for rs in [42,13,70,34,104]:
            each.append(np.load('./mae_rmse_test_experiences/'+score+'_scores/'+metric+str(rs)+'.npy'))
        m=np.mean(each,axis=0)
        std=np.std(each,axis=0)
        np.save('./mae_rmse_test_experiences/'+score+'_scores/'+metric+'_ave',m)
        np.save('./mae_rmse_test_experiences/' + score + '_scores/'+metric+'_std', std)