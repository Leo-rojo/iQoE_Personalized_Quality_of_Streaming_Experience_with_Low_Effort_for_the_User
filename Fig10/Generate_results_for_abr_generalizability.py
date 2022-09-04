from modAL.models import ActiveLearner,CommitteeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import pairwise_distances
import copy
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import os
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# AL strategies

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

def each_user(nr_chunks, rs, u, model, regr_choosen,n_queries,t_s):
    np.random.seed(42)
    abrs=['bb', 'th', 'mpc']
    print('start ' + str(nr_chunks) + '_' + str(rs) + '_' + str(u) + '_' + str(model) + '_' + str(t_s))
    # Constants
    nr_feat = nr_chunks * 10

    # all features
    synthetic_experiences = np.load('./exp_and_scores_each_ABR/7_chunks_exp_'+t_s+'.npy')
    scores_synthetic_users = np.load('./exp_and_scores_each_ABR/scores_scaled_'+t_s+'.npy')
    abrs.remove(t_s)
    abr_residual = abrs
    synth_exp1=np.load('./exp_and_scores_each_ABR/7_chunks_exp_'+abr_residual[0]+'.npy')
    scores_synth_1=np.load('./exp_and_scores_each_ABR/scores_scaled_'+abr_residual[0]+'.npy')
    synth_exp2 = np.load('./exp_and_scores_each_ABR/7_chunks_exp_' + abr_residual[1] + '.npy')
    scores_synth_2 = np.load('./exp_and_scores_each_ABR/scores_scaled_' + abr_residual[1] + '.npy')


    all_features = copy.copy(synthetic_experiences)
    users_scores = copy.copy(scores_synthetic_users)

    X = all_features
    y = scores_synthetic_users[u][model].reshape(500)
    X1= synth_exp1
    y1= scores_synth_1[u][model].reshape(500)
    X2 = synth_exp2
    y2 = scores_synth_2[u][model].reshape(500)
    model_name = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas'][model]

    # define min max scaler
    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=rs)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=rs)


    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_test1 = scaler.transform(X_test1)
    X_test2 = scaler.transform(X_test2)

    ###Active_leanring###
    n_initial = 1
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_init_training, y_init_training = X_train[initial_idx], np.array(y_train, dtype=int)[initial_idx]

    # Isolate the non-training examples we'll be querying.
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)

    regressor_gsio = ActiveLearner(
        estimator=xgb.XGBRegressor(n_estimators=100, max_depth=60, nthread=1),
        query_strategy=random_greedy_sampling_input_output,
        X_training=X_init_training.reshape(-1, nr_feat),
        y_training=y_init_training.reshape(-1, 1).flatten()
    )

    ##initial scores
    scores_gsio = [regressor_gsio.score(X_test, y_test)]
    scores_gsio1 = [regressor_gsio.score(X_test1, y_test1)]
    scores_gsio2 = [regressor_gsio.score(X_test2, y_test2)]

    ##initial lcc
    lccs_gsio = [pearsonr(regressor_gsio.predict(X_test), y_test)[0]]
    lccs_gsio1 = [pearsonr(regressor_gsio.predict(X_test1), y_test1)[0]]
    lccs_gsio2 = [pearsonr(regressor_gsio.predict(X_test2), y_test2)[0]]

    ##initial srocc
    srccs_gsio = [spearmanr(regressor_gsio.predict(X_test), y_test)[0]]
    srccs_gsio1 = [spearmanr(regressor_gsio.predict(X_test1), y_test1)[0]]
    srccs_gsio2 = [spearmanr(regressor_gsio.predict(X_test2), y_test2)[0]]

    ##initial knds
    knds_gsio = [kendalltau(regressor_gsio.predict(X_test), y_test)[0]]
    knds_gsio1 = [kendalltau(regressor_gsio.predict(X_test1), y_test1)[0]]
    knds_gsio2 = [kendalltau(regressor_gsio.predict(X_test2), y_test2)[0]]

    # initial maes
    maes_gsio = [mean_absolute_error(y_test, regressor_gsio.predict(X_test))]
    maes_gsio1 = [mean_absolute_error(y_test1, regressor_gsio.predict(X_test1))]
    maes_gsio2 = [mean_absolute_error(y_test2, regressor_gsio.predict(X_test2))]

    # initial rmse
    rmses_gsio = [sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test)))]
    rmses_gsio1 = [sqrt(mean_squared_error(y_test1, regressor_gsio.predict(X_test1)))]
    rmses_gsio2 = [sqrt(mean_squared_error(y_test2, regressor_gsio.predict(X_test2)))]

    X_pool_gsio = X_pool.copy()
    y_pool_gsio = y_pool.copy()

    # active learning
    tresh = 20
    count_queries = 1
    switch_bol = False
    for idx in range(n_queries):
        if count_queries > tresh:
            switch_bol = True
        # gsio
        query_idx, query_instance = regressor_gsio.query(X_pool_gsio,switch=switch_bol)
        # print('gs_' + str(query_idx))
        query_idx = int(query_idx)  # 0because it is a list in this particular case
        regressor_gsio.teach(np.array(X_pool_gsio[query_idx]).reshape(-1, nr_feat),
                             np.array(y_pool_gsio[query_idx]).reshape(-1, 1).flatten())
        X_pool_gsio, y_pool_gsio = np.delete(X_pool_gsio, query_idx, axis=0), np.delete(y_pool_gsio, query_idx)

        # save_queries scores
        scores_gsio.append(regressor_gsio.score(X_test, y_test))
        scores_gsio1.append(regressor_gsio.score(X_test1, y_test2))
        scores_gsio2.append(regressor_gsio.score(X_test2, y_test1))

        # save_queries lccs
        lccs_gsio.append(pearsonr(regressor_gsio.predict(X_test), y_test)[0])
        lccs_gsio1.append(pearsonr(regressor_gsio.predict(X_test1), y_test1)[0])
        lccs_gsio2.append(pearsonr(regressor_gsio.predict(X_test2), y_test2)[0])

        # save_queries rmse
        rmses_gsio.append(sqrt(mean_squared_error(y_test, regressor_gsio.predict(X_test))))
        rmses_gsio1.append(sqrt(mean_squared_error(y_test1, regressor_gsio.predict(X_test1))))
        rmses_gsio2.append(sqrt(mean_squared_error(y_test2, regressor_gsio.predict(X_test2))))

        ##save_queries srocc
        srccs_gsio.append(spearmanr(regressor_gsio.predict(X_test), y_test)[0])
        srccs_gsio1.append(spearmanr(regressor_gsio.predict(X_test1), y_test1)[0])
        srccs_gsio2.append(spearmanr(regressor_gsio.predict(X_test2), y_test2)[0])

        ##save_queries knds
        knds_gsio.append(kendalltau(regressor_gsio.predict(X_test), y_test)[0])
        knds_gsio1.append(kendalltau(regressor_gsio.predict(X_test1), y_test1)[0])
        knds_gsio2.append(kendalltau(regressor_gsio.predict(X_test2), y_test2)[0])

        # save_queries maes
        maes_gsio.append(mean_absolute_error(y_test, regressor_gsio.predict(X_test)))
        maes_gsio1.append(mean_absolute_error(y_test1, regressor_gsio.predict(X_test1)))
        maes_gsio2.append(mean_absolute_error(y_test2, regressor_gsio.predict(X_test2)))

        # print('training_query: '+str(count_queries))
        count_queries+=1

    # salve nelle folder shuffle
    # folders for metrics
    scores100 = [scores_gsio,scores_gsio1,scores_gsio2]
    lcc100 = [lccs_gsio,lccs_gsio1,lccs_gsio2]
    rmse100 = [rmses_gsio,rmses_gsio1,rmses_gsio2]
    maes100 = [maes_gsio,maes_gsio1,maes_gsio2]
    knd100 = [knds_gsio,knds_gsio1,knds_gsio2]
    srcc100 = [srccs_gsio,srccs_gsio1,srccs_gsio2]
    sco = [scores100, lcc100, rmse100, srcc100, maes100, knd100]
    conta = 0
    for met in ['R2', 'lcc', 'rmse', 'srcc', 'mae', 'knd']:
        main_path_save = regr_choosen + '_results_qn_' + str(n_queries) + '_nr_ch_' + str(nr_chunks) + '_' + str(
            n_initial)
        if not os.path.exists(
                main_path_save + '/' + model_name + '/user_' + str(u) + '/shuffle_' + str(rs) + '/' + met):
            os.makedirs(main_path_save + '/' + model_name + '/user_' + str(u) + '/ts_' + str(t_s) + '/shuffle_' + str(
                rs) + '/' + met)
        np.save(main_path_save + '/' + model_name + '/user_' + str(u) + '/ts_' + str(t_s) + '/shuffle_' + str(
            rs) + '/' + met + '/scores_for_ALstrat', sco[conta])  # salvo le 5 AL strategies
        conta += 1
    print('end ' + str(nr_chunks) + '_' + str(rs) + '_' + str(u) + '_' + str(model) + '_' + str(t_s))

if __name__ == "__main__":
    import time
    from multiprocessing import Pool

    # params
    comb_of_par = []
    reg = 'XGboost'  # ['RF','XGboost','SVR']
    n_queries = 60
    for nr_chunk in [7]:  # [2,4,8,
        for ts in ['bb','th','mpc']:
            for rs in [42, 13, 70, 34, 104]:
                for u in range(32):
                    for m in range(8):
                        model_name = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas'][m]
                        main_path = reg + '_results_qn_' + str(n_queries) + '_nr_ch_' + str(nr_chunk) + '_' + str(1)
                        if not os.path.exists(
                                main_path + '/' + model_name + '/user_' + str(u) + '/ts_' + str(ts) + '/shuffle_' + str(
                                        rs)):
                            comb_of_par.append((nr_chunk, rs, u, m, reg, n_queries, ts))
                            print(str(nr_chunk) + '_' + str(rs) + '_' + str(u) + '_' + str(m) + '_' + str(ts))
    print('param missing: ' + str(len(comb_of_par)))
    # else:
    # print(main_path + '/' + model_name + '/user_' + str(u) + '/rmses')

    time_1 = time.time()
    with Pool() as p:
        # p.map(each_user, [u for u in range(32)])
        p.starmap(each_user, comb_of_par)
    p.close()
    time_2 = time.time()
    time_interval = time_2 - time_1
    print(time_interval)


