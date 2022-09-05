import numpy as np
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVR
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, silhouette_samples, pairwise_distances_argmin_min
import xgboost as xgb
from modAL.models import ActiveLearner,CommitteeRegressor
from scipy.spatial.distance import jensenshannon
import itertools
from math import sqrt
from sklearn import linear_model

rs=42
numbers_of_groups=[1, 2, 4, 8, 16, 32, 64, 128, 256]
#funct
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
def find_group(user_choosen,splitted_group):
    for i in range(len(splitted_group)):
        if user_choosen in splitted_group[i]:
            return i

#collection
all_feat=np.load('./features_generated_experiences/feat_iQoE_for_synth_exp.npy')
all_scores=np.load('./synthetic_users_scores_for_generated_experiences/scaled/nrchunks_7.npy')
all_scores_ind_users=all_scores.reshape(32*8,1000)
#VA features
VA_feat=np.array(np.load('./features_generated_experiences/feat_videoAtlas_for_synth_exp.npy'))
V_feat=np.array(np.load('./features_generated_experiences/feat_vmaf_for_synth_exp.npy'))

#split train test mos and users to make the sorting based on 700 train and not all dataset
all_700_train=[]
for u in range(256):
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(VA_feat, all_scores_ind_users[u], test_size=0.3, random_state=rs)
    all_700_train.append(y_train_u)

#sort by jensen
mos_all=np.mean(all_700_train,axis=0)
js_distance_all=[jensenshannon(mos_all,i) for i in all_700_train]
index_sorted=np.argsort(js_distance_all)
#sort based on js_distance
sorted_scores_all_users=[all_scores_ind_users[i] for i in index_sorted]
sorted_all_700_train=[all_700_train[i] for i in index_sorted]
#####


#extract group1:[idx_user_general,worst_group,idx_relativetoworstgroup...26],group2:[...]
save_mos_700_for_split=[] #[1, 2, 4, 8, 16, 32, 64, 128, 256]
collect_groups_info=[]
for nr_group in numbers_of_groups:
    splitted_scores = list(split(sorted_all_700_train, nr_group))
    mos_of_splits = [np.mean(i, axis=0) for i in splitted_scores]
    save_mos_700_for_split.append(mos_of_splits)
    collect_by_group=[]
    for degree_of_worst in range(26):
        if nr_group==256:
            collect_by_group.append([collect_groups_info[0][degree_of_worst][0],collect_groups_info[0][degree_of_worst][2],collect_groups_info[0][degree_of_worst][2]])
        else:
            splitted_scores=list(split(sorted_all_700_train,nr_group))
            splitted_scores_idx = list(split(range(256), nr_group))
            #calcola mos per group
            mos_of_splits=[np.mean(i,axis=0) for i in splitted_scores]
            #worst user each splits
            worst_each_group=[]
            worst_value_each_group=[]

            save_all_jens=[]
            for idi,i in enumerate(splitted_scores):
                savek,saveik=[],[]
                for idk,k in enumerate(i):
                    savek.append(jensenshannon(mos_of_splits[idi],k))
                save_all_jens.append(savek)
            #worst of all
            save_all_jens_concat = list(itertools.chain.from_iterable(save_all_jens))
            idx_worst_user_in_general = np.argsort(save_all_jens_concat)[-(degree_of_worst+1)]
            for i in range(nr_group):
                if idx_worst_user_in_general in splitted_scores_idx[i]:
                    idx_worst_group=i
                    idx_worst_user_relative_group=splitted_scores_idx[i].index(idx_worst_user_in_general)
                    break
            collect_by_group.append([idx_worst_user_in_general,idx_worst_group,idx_worst_user_relative_group])
    collect_groups_info.append(collect_by_group)


#sorted_scores_all_user is the reference
tr_te_each_users=[]
for u in range(256):
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(VA_feat, sorted_scores_all_users[u], test_size=0.3, random_state=rs)
    tr_te_each_users.append([X_train_u,X_test_u,y_train_u,y_test_u])
#split train test mos and users for pQoE
tr_te_each_users_iqoe=[]
for u in range(256):
    X_train_u_iqoe, X_test_u_iqoe, y_train_u_iqoe, y_test_u_iqoe = train_test_split(all_feat,  sorted_scores_all_users[u], test_size=0.3, random_state=rs)
    tr_te_each_users_iqoe.append([X_train_u_iqoe,X_test_u_iqoe,y_train_u_iqoe,y_test_u_iqoe])
#split train test mos and users for v model
tr_te_each_users_vmodel=[]
for u in range(256):
    X_train_u_V, X_test_u_V, y_train_u_V, y_test_u_V = train_test_split(V_feat,  sorted_scores_all_users[u], test_size=0.3, random_state=rs)
    tr_te_each_users_vmodel.append([X_train_u_V,X_test_u_V,y_train_u_V,y_test_u_V])

#train va routine
def fit_supreg(all_features,mosscore):
    data = np.array(all_features)
    target = np.array(mosscore)

    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
                             param_grid={'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
                                         'gamma': np.logspace(-2, 2, 15)})
    regressor.fit(data, np.ravel(target))

    return regressor.best_estimator_
#train v routine
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

#train test mosV on collected group
save_for_gr_mae_V=[]
save_for_gr_rmse_V=[]
contagr=0
for groups in collect_groups_info:
    print('contag: '+str(contagr))
    save_26_mae_V=[]
    save_26_rmse_V=[]
    contus=0
    for userx in groups:
        print('contus: ' + str(contus))
        user_to_test=userx[0]
        group_to_train=userx[1]
        if os.path.isfile('./models_v/model_v_'+str(contagr)+'_'+str(group_to_train)+'.npy'):
            model_trained_u_V=np.load('./models_v/model_v_'+str(contagr)+'_'+str(group_to_train)+'.npy')
        else:
            model_trained_u_V=fit_linear(tr_te_each_users_vmodel[0][0], save_mos_700_for_split[contagr][group_to_train])#X_train_u,y_train_u
            np.save('./models_v/model_v_'+str(contagr)+'_'+str(group_to_train)+'.npy',model_trained_u_V)
        y_pred_test=[]
        for i in range(300):
            y_pred_test.append(np.dot(model_trained_u_V,tr_te_each_users_vmodel[user_to_test][1][i]))#np.dot(user_models[4], collect_sumvmaf[exp])
        mae=mean_absolute_error(tr_te_each_users_vmodel[user_to_test][3], y_pred_test)
        rmse=sqrt(mean_squared_error(tr_te_each_users_vmodel[user_to_test][3], y_pred_test))
        save_26_mae_V.append(mae)
        save_26_rmse_V.append(rmse)
        contus+=1
    save_for_gr_mae_V.append(save_26_mae_V)
    save_for_gr_rmse_V.append(save_26_rmse_V)
    contagr+=1
np.save('./result_jensen/mae_each_query_worst_users_V',save_for_gr_mae_V) #[9 groups][26 guys]
np.save('./result_jensen/rmse_each_query_worst_users_V', save_for_gr_rmse_V)






#train test mosvideoatlas on collected group
save_for_gr_mae=[]
save_for_gr_rmse=[]
contagr=0
for groups in collect_groups_info:
    print('contag: '+str(contagr))
    save_26_mae=[]
    save_26_rmse=[]
    contus=0
    for userx in groups:
        print('contus: ' + str(contus))
        user_to_test=userx[0]
        group_to_train=userx[1]
        if os.path.isfile('./models_va/model_va_'+str(contagr)+'_'+str(group_to_train)+'.pkl'):
            model_trained_u=pickle.load(open('./models_va/model_va_'+str(contagr)+'_'+str(group_to_train)+'.pkl', "rb"))
        else:
            model_trained_u=fit_supreg(tr_te_each_users[0][0], save_mos_700_for_split[contagr][group_to_train])#X_train_u,y_train_u
            pickle.dump(model_trained_u,open('./models_va/model_va_'+str(contagr)+'_'+str(group_to_train)+ '.pkl','wb'))
        y_pred_test=model_trained_u.predict(tr_te_each_users[user_to_test][1])
        mae=mean_absolute_error(tr_te_each_users[user_to_test][3], y_pred_test)
        rmse=sqrt(mean_squared_error(tr_te_each_users[user_to_test][3], y_pred_test))
        save_26_mae.append(mae)
        save_26_rmse.append(rmse)
        contus+=1
    save_for_gr_mae.append(save_26_mae)
    save_for_gr_rmse.append(save_26_rmse)
    contagr+=1
np.save('./result_jensen/mae_each_query_worst_users_va',save_for_gr_mae) #[9 groups][26 guys]
np.save('./result_jensen/rmse_each_query_worst_users_va', save_for_gr_rmse)

#iGS
q_fix=49
iqoesave_for_gr_mae=[]
iqoesave_for_gr_rmse=[]
contagr=0
for groups in collect_groups_info:
    print('contagiqoe: '+str(contagr))
    iqoesave_26_mae=[]
    iqoesave_26_rmse=[]
    contus=0
    for userx in groups:

        np.random.seed(42)
        nr_feat=70 #nr chunksXfeatures

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
        ###Active_leanring### tr_te_each_users_iqoe.append([X_train_u_iqoe,X_test_u_iqoe,y_train_u_iqoe,y_test_u_iqoe])
        myuser=tr_te_each_users_iqoe[userx[0]]
        X_train = myuser[0]
        X_test = myuser[1]
        y_train = myuser[2]
        y_test = myuser[3]
        if not os.path.isfile('./models_iQoE/model_iQoE_'+str(userx[0])+'_'+str(q_fix)+'.json'):
            n_initial = 1
            initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
            X_init_training, y_init_training = X_train[initial_idx], np.array(y_train,dtype=int)[initial_idx]

            # Isolate the non-training examples we'll be querying.
            X_pool = np.delete(X_train, initial_idx, axis=0)
            y_pool = np.delete(y_train, initial_idx, axis=0)

            regressor_gsio = ActiveLearner(
                estimator=xgb.XGBRegressor(n_estimators = 100, max_depth = 60,nthread=1),
                query_strategy=random_greedy_sampling_input_output,
                X_training=X_init_training.reshape(-1, nr_feat),
                y_training=y_init_training.reshape(-1, 1).flatten()
            )
            #maes_gsio = [mean_absolute_error(y_test, regressor_gsio.predict(X_test))]
            #rmses_gsio = [sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test)))]
            X_pool_gsio=X_pool.copy()
            y_pool_gsio=y_pool.copy()

            # active learning
            n_queries=60
            t_s = 20
            count_queries = 1
            switch_bol = False
            for idx in range(n_queries):
                if count_queries > t_s:
                    switch_bol = True
                # gsio
                query_idx, query_instance = regressor_gsio.query(X_pool_gsio, switch=switch_bol)
                # print('gs_' + str(query_idx))
                query_idx = int(query_idx)  # 0because it is a list in this particular case
                regressor_gsio.teach(np.array(X_pool_gsio[query_idx]).reshape(-1, nr_feat),
                                   np.array(y_pool_gsio[query_idx]).reshape(-1, 1).flatten())
                X_pool_gsio, y_pool_gsio = np.delete(X_pool_gsio, query_idx, axis=0), np.delete(y_pool_gsio, query_idx)

                print('training_query: ' + str(count_queries))
                count_queries += 1
                if idx in [49]:#, 99, 149, 179, 199]:
                    regressor_gsio.estimator.save_model('./models_iQoE/model_iQoE_'+str(userx[0])+'_'+str(idx)+'.json')

        regressor_gsio = xgb.XGBRegressor()
        regressor_gsio.load_model('./models_iQoE/model_iQoE_' + str(userx[0]) + '_' + str(q_fix) + '.json')
        #save_queries maes
        iqoesave_26_mae.append(mean_absolute_error(y_test, regressor_gsio.predict(X_test)))
        iqoesave_26_rmse.append(sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test))))
        contus+=1
    iqoesave_for_gr_mae.append(iqoesave_26_mae)
    iqoesave_for_gr_rmse.append(iqoesave_26_rmse)
    contagr+=1
np.save('./result_jensen/mae_'+str(q_fix)+'_worst_users_iqoe',iqoesave_for_gr_mae)
np.save('./result_jensen/rmse_'+str(q_fix)+'_worst_users_iqoe', iqoesave_for_gr_rmse)

print('done')


