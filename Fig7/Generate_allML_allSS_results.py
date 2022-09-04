import os
from modAL.models import ActiveLearner,CommitteeRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples, pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.base import clone
import pandas as pd
import copy
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import os
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

# AL strategies
def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]
def greedy_sampling_input(regressor, X_pool):
    dist_matrix = pairwise_distances(regressor.X_training, X_pool)
    dist_to_training_set = np.amin(dist_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X_pool[query_idx]
def cluster_uncertainty(regressor, X_pool, n_c=7, n_instances=1):
    query_idx = []
    kmeans = KMeans(n_c)
    y_pool = pd.DataFrame(regressor.predict(X_pool), columns=['y'])
    kmeans.fit(X_pool)
    y_pool['cluster'] = kmeans.labels_
    y_pool['silhouette'] = silhouette_samples(y_pool['y'].to_numpy().reshape(-1, 1), y_pool['cluster'])
    selected_clusters = y_pool.groupby('cluster').agg({'y': 'var'}).nlargest(n_instances, 'y').index.tolist()
    for cluster in selected_clusters:
        query_idx.append(y_pool[y_pool['cluster'] == cluster]['silhouette'].idxmin())
    return query_idx
def random_greedy_sampling_input_output(regressor, X): #it is iGS
    y = regressor.predict(X)
    dist_x_matrix = pairwise_distances(regressor.X_training, X)
    dist_y_matrix = pairwise_distances(
        regressor.y_training.reshape(-1, 1), y.reshape(-1, 1)
    )
    dist_to_training_set = np.amin(dist_x_matrix * dist_y_matrix, axis=0)
    query_idx = np.argmax(dist_to_training_set)
    return query_idx, X[query_idx]
def committ_qs(committee, X_pool):  # Pool-Based Sequential Active Learning for Regression Dongrui Wu, qbc with variance taken from other paper
    # print(X_pool)
    variances = []
    vote_learners = committee.vote(X_pool)  # obtain prediction of various learners
    for i in vote_learners:
        mean = np.mean(i)
        s = 0
        for k in range(len(committee)):
            s += (i[k] - mean) ** 2
        s = s / len(committee)
        variances.append(s)
    # print(variances)
    query_idx = np.argmax(variances)
    return query_idx, X_pool[query_idx]

def each_user(nr_chunks, rs, u, model, regr_choosen,n_queries):
    np.random.seed(42)
    print('start ' + str(nr_chunks)+'_'+str(rs)+'_'+str(u)+'_'+str(model))
    # Constants
    nr_feat = nr_chunks * 10

    # all features
    synthetic_experiences = np.load('./features_generated_experiences/feat_iQoE_for_synth_exp.npy')
    scores_synthetic_users = np.load('./synthetic_users_scores_for_generated_experiences/scaled/nrchunks_7.npy')

    all_features = copy.copy(synthetic_experiences)
    users_scores = copy.copy(scores_synthetic_users)

    X = all_features
    y = scores_synthetic_users[u][model].reshape(1000)
    model_name = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas'][model]

    # define min max scaler
    scaler = MinMaxScaler()


    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=rs)

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ###Active_leanring###
    n_initial = 1
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_init_training, y_init_training = X_train[initial_idx], np.array(y_train,dtype=int)[initial_idx]

    # Isolate the non-training examples we'll be querying.
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)

    Regressors_considered=[RandomForestRegressor(n_estimators = 50, max_depth = 60)]
    Regressors_considered.append(xgb.XGBRegressor(n_estimators = 100, max_depth = 60,nthread=1))
    Regressors_considered.append(sklearn.svm.SVR(kernel = 'rbf', gamma= 0.5, C= 100))
    Regressors_considered.append(GaussianProcessRegressor(kernel=RationalQuadratic()+2,alpha=5))

    regr_choosen_idx=['RF', 'XGboost', 'SVR','GP'].index(regr_choosen)
    regr_1 = Regressors_considered[regr_choosen_idx]

    regressor_random = ActiveLearner(
        estimator=regr_1,
        query_strategy=random_sampling,
        X_training=X_init_training.reshape(-1,nr_feat), y_training=y_init_training.reshape(-1,1).flatten()
    )

    regr_2 = clone(regr_1)
    regressor_cluster = ActiveLearner(
        estimator=regr_2,
        query_strategy=cluster_uncertainty,
        X_training=X_init_training.reshape(-1, nr_feat), y_training=y_init_training.reshape(-1, 1).flatten()
    )

    regr_3 = clone(regr_1)
    regressor_gs = ActiveLearner(
        estimator=regr_3,
        query_strategy=greedy_sampling_input,
        X_training=X_init_training.reshape(-1, nr_feat), y_training=y_init_training.reshape(-1, 1).flatten()
    )

    ###
    # initializing Committee members
    n_members = 3
    learner_list = list()
    # a list of ActiveLearners:
    for member_idx in range(n_members):

        # initializing learner
        learner = ActiveLearner(
            estimator=clone(regr_1),
            X_training=X_init_training.reshape(-1, nr_feat), y_training=y_init_training.reshape(-1, 1).flatten()
        )
        learner_list.append(learner)

    # inp output greedy sampling
    regressor_comm = CommitteeRegressor(learner_list=learner_list, query_strategy=committ_qs)

    regr_5 = clone(regr_1)
    regressor_gsio = ActiveLearner(
        estimator=regr_5,
        query_strategy=random_greedy_sampling_input_output,
        X_training=X_init_training.reshape(-1, nr_feat),
        y_training=y_init_training.reshape(-1, 1).flatten()
    )

    ##initial scores
    scores_r = [regressor_random.score(X_test, y_test)]
    scores_cluster = [regressor_cluster.score(X_test, y_test)]
    scores_gs = [regressor_gs.score(X_test, y_test)]
    scores_comm = [r2_score(y_test, regressor_comm.predict(X_test))]
    scores_gsio = [regressor_gsio.score(X_test, y_test)]

    ##initial lcc
    lccs_r = [pearsonr(regressor_random.predict(X_test), y_test)[0]]
    lccs_cluster =[pearsonr(regressor_cluster.predict(X_test), y_test)[0]]
    lccs_gs = [pearsonr(regressor_gs.predict(X_test), y_test)[0]]
    lccs_comm = [pearsonr(regressor_comm.predict(X_test), y_test)[0]]
    lccs_gsio = [pearsonr(regressor_gsio.predict(X_test), y_test)[0]]

    ##initial srocc
    srccs_r = [spearmanr(regressor_random.predict(X_test), y_test)[0]]
    srccs_cluster = [spearmanr(regressor_cluster.predict(X_test), y_test)[0]]
    srccs_gs = [spearmanr(regressor_gs.predict(X_test), y_test)[0]]
    srccs_comm = [spearmanr(regressor_comm.predict(X_test), y_test)[0]]
    srccs_gsio = [spearmanr(regressor_gsio.predict(X_test), y_test)[0]]

    ##initial knds
    knds_r = [kendalltau(regressor_random.predict(X_test), y_test)[0]]
    knds_cluster = [kendalltau(regressor_cluster.predict(X_test), y_test)[0]]
    knds_gs = [kendalltau(regressor_gs.predict(X_test), y_test)[0]]
    knds_comm = [kendalltau(regressor_comm.predict(X_test), y_test)[0]]
    knds_gsio = [kendalltau(regressor_gsio.predict(X_test), y_test)[0]]

    # initial maes
    maes_r = [mean_absolute_error(y_test, regressor_random.predict(X_test))]
    maes_cluster = [mean_absolute_error(y_test, regressor_cluster.predict(X_test))]
    maes_gs = [mean_absolute_error(y_test, regressor_gs.predict(X_test))]
    maes_comm = [mean_absolute_error(y_test, regressor_comm.predict(X_test))]
    maes_gsio = [mean_absolute_error(y_test, regressor_gsio.predict(X_test))]

    #initial rmse
    rmses_r = [sqrt(mean_squared_error(y_test, regressor_random.predict(X_test)))]
    rmses_cluster = [sqrt(mean_squared_error(y_test,regressor_cluster.predict(X_test)))]
    rmses_gs = [sqrt(mean_squared_error(y_test, regressor_gs.predict(X_test)))]
    rmses_comm = [sqrt(mean_squared_error(y_test, regressor_comm.predict(X_test)))]
    rmses_gsio = [sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test)))]

    X_pool_random,X_pool_cluster,X_pool_gs,X_pool_comm,X_pool_gsio=X_pool.copy(),X_pool.copy(),X_pool.copy(),X_pool.copy(),X_pool.copy()
    y_pool_random,y_pool_cluster,y_pool_gs,y_pool_comm,y_pool_gsio=y_pool.copy(),y_pool.copy(),y_pool.copy(),y_pool.copy(),y_pool.copy()

    # active learning
    t_s=20
    count_queries=1
    for idx in range(n_queries):

        #take random queries
        if count_queries<t_s:
            n_samples = len(X_pool_random)
            query_idx = np.random.choice(range(n_samples))
        #random
        if count_queries>t_s:
            query_idx, query_instance = regressor_random.query(X_pool_random)
        query_idx = int(query_idx)
        regressor_random.teach(np.array(X_pool_random[query_idx]).reshape(-1,nr_feat), np.array(y_pool_random[query_idx]).reshape(-1,1).flatten())
        X_pool_random, y_pool_random = np.delete(X_pool_random, query_idx, axis=0), np.delete(y_pool_random, query_idx)


        #cluster
        if count_queries > t_s:
            query_idx, query_instance = regressor_cluster.query(X_pool_cluster,n_c=7,n_instances=1)
            query_idx=int(query_idx[0])
        else:
           query_idx = int(query_idx)
        regressor_cluster.teach(np.array(X_pool_cluster[query_idx]).reshape(-1, nr_feat), np.array(y_pool_cluster[query_idx]).reshape(-1, 1).flatten())
        X_pool_cluster, y_pool_cluster = np.delete(X_pool_cluster, query_idx, axis=0), np.delete(y_pool_cluster, query_idx)

        #gs
        if count_queries > t_s:
            query_idx, query_instance = regressor_gs.query(X_pool_gs)
        #print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        regressor_gs.teach(np.array(X_pool_gs[query_idx]).reshape(-1, nr_feat),
                           np.array(y_pool_gs[query_idx]).reshape(-1, 1).flatten())
        X_pool_gs, y_pool_gs = np.delete(X_pool_gs, query_idx, axis=0), np.delete(y_pool_gs,query_idx)


        #committee
        if count_queries > t_s:
            query_idx, query_instance = regressor_comm.query(X_pool_comm)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)
        regressor_comm.teach(np.array(X_pool_comm[query_idx]).reshape(-1, nr_feat),
                           np.array(y_pool_comm[query_idx]).reshape(-1, 1).flatten())
        X_pool_comm, y_pool_comm = np.delete(X_pool_comm, query_idx, axis=0), np.delete(y_pool_comm, query_idx)


        # gsio
        if count_queries > t_s:
            query_idx, query_instance = regressor_gsio.query(X_pool_gsio)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        regressor_gsio.teach(np.array(X_pool_gsio[query_idx]).reshape(-1, nr_feat),
                           np.array(y_pool_gsio[query_idx]).reshape(-1, 1).flatten())
        X_pool_gsio, y_pool_gsio = np.delete(X_pool_gsio, query_idx, axis=0), np.delete(y_pool_gsio, query_idx)



        #save_queries scores
        scores_r.append(regressor_random.score(X_test,y_test))
        scores_cluster.append(regressor_cluster.score(X_test, y_test))
        scores_gs.append(regressor_gs.score(X_test,y_test))
        scores_comm.append(r2_score(y_test, regressor_comm.predict(X_test)))
        scores_gsio.append(regressor_gsio.score(X_test, y_test))

        # save_queries lccs
        lccs_r.append(pearsonr(regressor_random.predict(X_test), y_test)[0])
        lccs_cluster.append(pearsonr(regressor_cluster.predict(X_test), y_test)[0])
        lccs_gs.append(pearsonr(regressor_gs.predict(X_test), y_test)[0])
        lccs_comm.append(pearsonr(regressor_comm.predict(X_test), y_test)[0])
        lccs_gsio.append(pearsonr(regressor_gsio.predict(X_test), y_test)[0])

        # save_queries rmse
        rmses_r.append(sqrt(mean_squared_error(y_test, regressor_random.predict(X_test))))
        rmses_cluster.append(sqrt(mean_squared_error(y_test, regressor_cluster.predict(X_test))))
        rmses_gs.append(sqrt(mean_squared_error(y_test, regressor_gs.predict(X_test))))
        rmses_comm.append(sqrt(mean_squared_error(y_test, regressor_comm.predict(X_test))))
        rmses_gsio.append(sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test))))

        ##save_queries srocc
        srccs_r.append(spearmanr(regressor_random.predict(X_test), y_test)[0])
        srccs_cluster.append(spearmanr(regressor_cluster.predict(X_test), y_test)[0])
        srccs_gs.append(spearmanr(regressor_gs.predict(X_test), y_test)[0])
        srccs_comm.append(spearmanr(regressor_comm.predict(X_test), y_test)[0])
        srccs_gsio.append(spearmanr(regressor_gsio.predict(X_test), y_test)[0])

        ##save_queries knds
        knds_r.append(kendalltau(regressor_random.predict(X_test), y_test)[0])
        knds_cluster.append(kendalltau(regressor_cluster.predict(X_test), y_test)[0])
        knds_gs.append(kendalltau(regressor_gs.predict(X_test), y_test)[0])
        knds_comm.append(kendalltau(regressor_comm.predict(X_test), y_test)[0])
        knds_gsio.append(kendalltau(regressor_gsio.predict(X_test), y_test)[0])

        #save_queries maes
        maes_r.append(mean_absolute_error(y_test, regressor_random.predict(X_test)))
        maes_cluster.append(mean_absolute_error(y_test, regressor_cluster.predict(X_test)))
        maes_gs.append(mean_absolute_error(y_test, regressor_gs.predict(X_test)))
        maes_comm.append(mean_absolute_error(y_test, regressor_comm.predict(X_test)))
        maes_gsio.append(mean_absolute_error(y_test, regressor_gsio.predict(X_test)))

        #print('training_query: '+str(count_queries))
        count_queries+=1

    #salve nelle folder shuffle
    # folders for metrics
    scores100=[scores_r,scores_cluster,scores_gs,scores_comm,scores_gsio]
    lcc100=[lccs_r,lccs_cluster,lccs_gs,lccs_comm,lccs_gsio]
    rmse100=[rmses_r,rmses_cluster,rmses_gs,rmses_comm,rmses_gsio]
    maes100=[maes_r,maes_cluster,maes_gs,maes_comm,maes_gsio]
    knd100=[knds_r,knds_cluster,knds_gs,knds_comm,knds_gsio]
    srcc100=[srccs_r,srccs_cluster,srccs_gs,srccs_comm,srccs_gsio]
    sco=[scores100,lcc100,rmse100,srcc100,maes100,knd100]
    conta=0
    for met in ['R2', 'lcc', 'rmse', 'srcc', 'mae', 'knd']:
        main_path_save = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_' + str(nr_chunks)+'_'+str(n_initial)
        if not os.path.exists(main_path_save + '/' + model_name + '/user_' + str(u) +'/shuffle_'+str(rs)+'/'+met):
            os.makedirs(main_path_save + '/' + model_name + '/user_' + str(u) +'/shuffle_'+str(rs)+'/'+met)
        np.save(main_path_save + '/' + model_name + '/user_' + str(u) + '/shuffle_'+str(rs)+ '/'+met+'/scores_for_ALstrat', sco[conta]) #salvo le 5 AL strategies
        conta+=1
    print('end ' + str(nr_chunks)+'_'+str(rs)+'_'+str(u)+'_'+str(model))

if __name__ == "__main__":
    from multiprocessing import Pool
    nr_chunk=7

    # params
    comb_of_par = []
    reg='XGboost'#['RF','XGboost','SVR']
    n_queries = 250

    for reg in ['RF','XGboost','SVR','GP']:
        for rs in [42,13,70,34,104]:
                for u in range(32):
                    for m in range(8):
                        model_name=['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas'][m]
                        main_path =reg + '_results_qn_' + str(n_queries) + '_nr_ch_' + str(nr_chunk)+'_'+str(1)
                        if not os.path.exists(main_path + '/' + model_name + '/user_' + str(u) + '/shuffle_'+str(rs)):
                            comb_of_par.append((nr_chunk, rs, u, m, reg, n_queries))
                            print(str(nr_chunk)+ '_' + str(rs)+'_' + str(u) +'_'+ str(m))
    print('param missing: '+ str(len(comb_of_par)))
                #else:
                    #print(main_path + '/' + model_name + '/user_' + str(u) + '/rmses')
    with Pool() as p:
        #p.map(each_user, [u for u in range(32)])
        p.starmap(each_user, comb_of_par)
    p.close()
    print('done')


