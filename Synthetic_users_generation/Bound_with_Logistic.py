import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit,minimize
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import os

models_folder='Fitted_models_without_logistic'
real_scores=np.load('./users_scores_hdtv.npy')
models=['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']

#collect all features w4 for each model
all_features=[]
for i in models:
    collect_temp = []
    all_features.append(np.load('./features_for_synthetic_user_fitting/feat_'+i+'.npy'))

#def sigmoid_scaling(xdata,ydata):  #xdata are the output of my model, ydata are real ones
def sigmoid_scaling(xdata,ydata,u):
    initialx0=np.median(xdata)
    def sigmoid(x, k, x0):
        return (99.0 / (1 + np.exp(-k * (x - x0))))+1

    # Fit the curve
    popt, pcov = curve_fit(sigmoid, xdata, ydata,p0=[1,initialx0],maxfev=20000)
    estimated_k, estimated_x0 = popt

    y_fitted = sigmoid(xdata, k=estimated_k, x0=estimated_x0)
    return y_fitted,estimated_k,estimated_x0

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


all_scores_no_scaled=[] #32*8
all_scores_sigmoid_scaled=[]
params_sigmoid=[]
scores_by_users_not_scaled=[]
for u in range(32):
    print(u)

    scores_by_users_scaled_sigmoid = []
    params_sigmoid_user=[]
    user_models=all_synthetic_users[u]
    for kind_of_models in models:
        if kind_of_models=='bit':#[s_bit,s_dif_bit,s_psnr,s_dif_psnr,s_ssim,s_dif_ssim,s_vmaf,s_dif_vmaf,s_bit_log,s_dif_bit_log,ave_st_FTW,nr_stall_FTW,s_reb]
            temp_score=[]
            for exp in range(len(all_features[0])):
                score=np.dot(user_models[0],all_features[0][exp]) #here should go the non linear mapping eventually for real context.
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))

            # map logistic
            qoe_model_outputs=temp_score
            qoe_scaled, alpha, beta = sigmoid_scaling(qoe_model_outputs, real_scores[u],u)
            scores_by_users_scaled_sigmoid.append(qoe_scaled) #min max scalere here but it needs all the scores given in order to map them.
            params_sigmoid_user.append([alpha,beta])
        elif kind_of_models=='logbit':
            temp_score = []
            for exp in range(len(all_features[0])):
                score=np.dot(user_models[1], all_features[1][exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            # map logistic
            qoe_model_outputs = temp_score
            qoe_scaled, alpha, beta = sigmoid_scaling(qoe_model_outputs, real_scores[u],u)
            scores_by_users_scaled_sigmoid.append(qoe_scaled)  # min max scalere here but it needs all the scores given in order to map them.
            params_sigmoid_user.append([alpha, beta])
        elif kind_of_models=='psnr':
            temp_score = []
            for exp in range(len(all_features[0])):
                score=np.dot(user_models[2], all_features[2][exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            # map logistic
            qoe_model_outputs = temp_score
            qoe_scaled, alpha, beta = sigmoid_scaling(qoe_model_outputs, real_scores[u],u)
            scores_by_users_scaled_sigmoid.append(qoe_scaled)  # min max scalere here but it needs all the scores given in order to map them.
            params_sigmoid_user.append([alpha, beta])
        elif kind_of_models=='ssim':
            temp_score = []
            for exp in range(len(all_features[0])):
                score = np.dot(user_models[3],  all_features[3][exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            # map logistic
            qoe_model_outputs = temp_score
            qoe_scaled, alpha, beta = sigmoid_scaling(qoe_model_outputs, real_scores[u],u)
            scores_by_users_scaled_sigmoid.append(qoe_scaled)  # min max scalere here but it needs all the scores given in order to map them.
            params_sigmoid_user.append([alpha, beta])
        elif kind_of_models=='vmaf':
            temp_score = []
            for exp in range(len(all_features[0])):
                score = np.dot(user_models[4], all_features[4][exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            # map logistic
            qoe_model_outputs = temp_score
            qoe_scaled, alpha, beta = sigmoid_scaling(qoe_model_outputs, real_scores[u],u)
            scores_by_users_scaled_sigmoid.append(qoe_scaled)  # min max scalere here but it needs all the scores given in order to map them.
            params_sigmoid_user.append([alpha, beta])
        elif kind_of_models=='FTW':
            temp_score = []
            for exp in range(len(all_features[0])):
                a, b, c, d = user_models[5]
                x1, x2 = all_features[5][exp]
                score = a * np.exp(-(b * x1 + c) * x2) + d
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            # map logistic
            qoe_model_outputs = temp_score
            qoe_scaled, alpha, beta = sigmoid_scaling(qoe_model_outputs, real_scores[u],u)
            scores_by_users_scaled_sigmoid.append(qoe_scaled)  # min max scalere here but it needs all the scores given in order to map them.
            params_sigmoid_user.append([alpha, beta])
        elif kind_of_models=='SDNdash':
            temp_score = []
            for exp in range(len(all_features[0])):
                score = np.dot(user_models[6], all_features[6][exp])
                temp_score.append(score)
            scores_by_users_not_scaled.append((np.array(temp_score).reshape(-1, 1)))
            # map logistic
            qoe_model_outputs = temp_score
            qoe_scaled, alpha, beta = sigmoid_scaling(qoe_model_outputs, real_scores[u],u)
            scores_by_users_scaled_sigmoid.append(qoe_scaled)  # min max scalere here but it needs all the scores given in order to map them.
            params_sigmoid_user.append([alpha, beta])
        elif kind_of_models== 'videoAtlas':
            with open('./' + models_folder + '/organized_by_users/user_' + str(u) + '/model_videoAtlas.pkl','rb') as handle:
                pickled_atlas=pickle.load(handle)
            videoAtlasregressor=pickled_atlas #0 there is the mdoel,
            score = videoAtlasregressor.predict(all_features[7])
            scores_by_users_not_scaled.append((np.array(score).reshape(-1,1)))
            # map logistic
            qoe_model_outputs = temp_score
            qoe_scaled, alpha, beta = sigmoid_scaling(qoe_model_outputs, real_scores[u],u)
            scores_by_users_scaled_sigmoid.append(qoe_scaled)  # min max scalere here but it needs all the scores given in order to map them.
            params_sigmoid_user.append([alpha, beta])

    all_scores_no_scaled.append(scores_by_users_not_scaled)
    all_scores_sigmoid_scaled.append(scores_by_users_scaled_sigmoid)
    params_sigmoid.append(params_sigmoid_user)
np.save('./save_param_sigmoids/params_sigmoid.npy',params_sigmoid)
print('done')
