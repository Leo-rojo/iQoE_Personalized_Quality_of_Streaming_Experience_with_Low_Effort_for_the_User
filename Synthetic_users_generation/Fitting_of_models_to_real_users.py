import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit,minimize
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import os
collect_all=[]
users_scores=np.load('./users_scores_hdtv.npy')
l=['bit','logbit','psnr','ssim','vmaf','FTW','SDNdash','videoAtlas']

def fit_linear(all_features,users_scores,u):
    # multi-linear model fitting
    X = all_features
    y = users_scores

    lm = linear_model.LinearRegression(fit_intercept=False)
    model = lm.fit(X, y[u])

    alpha = lm.coef_[0]
    beta = lm.coef_[1]
    gamma = lm.coef_[2]

    return [alpha, beta, gamma]

def fit_nonlinear(all_features,users_scores,u):
    def fun(data,a, b, c, d):
        x1, x2 = data
        y = a * np.exp(-(b * x1 + c) * x2) + d
        return y

    # Fit the curve
    popt, pcov = curve_fit(fun, all_features, users_scores[u], maxfev=1000000)
    estimated_a, estimated_b, estimated_c, estimated_d = popt
    return estimated_a, estimated_b, estimated_c, estimated_d

def fit_supreg(all_features,users_scores,u):
    data = np.array(all_features)
    target = np.array(users_scores[u])

    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
                             param_grid={'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
                                         'gamma': np.logspace(-2, 2, 15)})
    regressor.fit(data, np.ravel(target))

    return regressor.best_estimator_


for user in range(32):
    if not os.path.exists('./Fitted_models_without_logistic/organized_by_users/user_'+str(user)):
        os.makedirs('./Fitted_models_without_logistic/organized_by_users/user_'+str(user))
for i in range(len(l)):
    if not os.path.exists('./Fitted_models_without_logistic/organized_by_type/models_'+l[i]):
        os.makedirs('./Fitted_models_without_logistic/organized_by_type/models_'+l[i])

for i in l:
    collect_temp = []
    all_features=np.load('./features_for_synthetic_user_fitting/feat_'+i+'.npy')

    if i=='FTW':
        for u in range(len(users_scores)):
            collect_temp.append(fit_nonlinear((all_features[:,0],all_features[:,1]), users_scores, u))
    elif i=='videoAtlas':
        for u in range(len(users_scores)):
            pickle.dump(fit_supreg(all_features,users_scores,u), open('./Fitted_models_without_logistic/organized_by_users/user_' + str(u)+'/model_'+ str(i) + '.pkl', 'wb'))
            collect_temp.append('videoAtlas_user' + str(u))
    else:
        for u in range(len(users_scores)):
            collect_temp.append(fit_linear(all_features, users_scores, u))

    collect_all.append(collect_temp)
    print(i)


#save models for type
for i in range(len(l)):
    np.save('./Fitted_models_without_logistic/organized_by_type/models_'+l[i],collect_all[i])

#save models for each users
for user in range(32):
    for i in range(len(l)):
        np.save('./Fitted_models_without_logistic/organized_by_users/user_' + str(user)+'/model_'+str(l[i]),collect_all[i][user])

print('done')






