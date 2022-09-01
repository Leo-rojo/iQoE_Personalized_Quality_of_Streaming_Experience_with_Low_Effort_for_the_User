from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.font_manager as font_manager
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#need to be in folder c

#remove type3 font
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#fix seed for reproducibility
rs=42
#take mos scores from original hdtv scores
xls = pd.ExcelFile(r"../hdtv_scores.xlsx") #user before absolute file path
sheetX = xls.parse(0)
mosarray=sheetX['mos'].tolist() # da splittare in 70-30

#calculate min and max of all the users scores
y2=[]
for u in range(1,33):
    y=sheetX['User '+str(u)].tolist()
    y2=y2+y
    totusmin=min(y)
    totusmax=max(y)

#take all individual user scores which I have saved previously
collect_all=[]
users_scores=np.load('../users_scores_hdtv.npy')

#function to fit video_atlas
def fit_supreg(all_features,mosscore):
    data = np.array(all_features)
    target = np.array(mosscore)

    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
                             param_grid={'C': [1e-1, 1e0, 1e1, 1e2, 1e3],
                                         'gamma': np.logspace(-2, 2, 15)})
    regressor.fit(data, np.ravel(target))

    return regressor.best_estimator_

#load all videoatlas features
all_features = np.load('./feat_videoAtlas.npy')

#split train test mos and users
X_train_mos, X_test_mos, y_train_mos, y_test_mos = train_test_split(all_features, mosarray, test_size=0.3, random_state=rs)
tr_te_each_users=[]
for u in range(32):
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(all_features, users_scores[u], test_size=0.3, random_state=rs)
    tr_te_each_users.append([X_train_u,X_test_u,y_train_u,y_test_u])


#train videoatlas with mos scores
model_trained_mos=fit_supreg(X_train_mos,y_train_mos)
pickle.dump(model_trained_mos, open('./model_'+ str('mosvideoatlas') + '.pkl', 'wb'))

#train personalized videoatlas for each user
save_trained_model_each_users=[]
for u in range(32):
    model_trained_u=fit_supreg(tr_te_each_users[u][0], tr_te_each_users[u][2])#X_train_u,y_train_u
    save_trained_model_each_users.append(model_trained_u)
    pickle.dump(model_trained_u, open('./personal_videoatlas_users/model_user'+str(u) + '.pkl', 'wb'))
    print('model_user'+str(u)+'_done')

########### it takes some time##################

#load the models (useful if you don't want to rerun the training every time)
with open('model_'+ str('mosvideoatlas') + '.pkl', 'rb') as f:
    model_trained_mos = pickle.load(f)
save_trained_model_each_users=[]
for u in range(32):
    with open('personal_videoatlas_users/model_user'+str(u) + '.pkl', 'rb') as f:
        save_trained_model_each_users.append(pickle.load(f))

#predict with mosmodel each user test set they are all the same but I do it anyhow just to confirm that everything is working properly
mosmodel_us_scores=[]
for u in range(32):
    user_u_scores=model_trained_mos.predict(tr_te_each_users[u][1])#X_test_u
    mosmodel_us_scores.append(user_u_scores)
#predict with each individual model the user test set
save_us_scores=[]
for u in range(32):
    user_u_scores=save_trained_model_each_users[u].predict(tr_te_each_users[u][1])#X_test_u
    save_us_scores.append(user_u_scores)

#####boxplots plot

#considered users
user_considered=[26,8,1,29]

#considered user position once sorted
user_considered_order=[1,2,31,32]

#users name for dataframe structure
users=[]
k=0
mosscores=mosmodel_us_scores[0]
for u in user_considered:
    for i in range(len(mosscores)):
        users.append('H'+str(user_considered_order[k]))
    k+=1

#mos score vector
mosscoresplot=[]
for u in range(len(user_considered)):
    mosscoresplot=mosscoresplot+mosscores.tolist()

#personal score vector
personalplot=[]
for u in user_considered:
    personalplot=personalplot+save_us_scores[u-1].tolist()

#real scores vector
realplot=[]
for u in user_considered:
    realplot=realplot+tr_te_each_users[u-1][3].tolist() #y_test_u for each users

#put in dataframe e format for seaborn plot
df = pd.DataFrame({'Users':users,\
                  'Personalized':personalplot,'Group-based':mosscoresplot,'SA':realplot})
df = df[['Users','SA','Group-based','Personalized']]


####################plot
fig1=plt.figure(figsize=(20, 10), dpi=100)
sns.set(rc={'figure.figsize':(8,3)},style="white")
dd=pd.melt(df,id_vars=['Users'],value_vars=['SA','Group-based','Personalized'],var_name='')
my_colors = ["#ffd700", "#4b8521", "#fe0000",]
sns.set_palette(my_colors)
ax=sns.boxplot(x='Users',y='value',data=dd,hue='')
font = font_manager.FontProperties(size=50) #per font legenda
ax.tick_params(bottom=True) #add ticks down
ax.tick_params(left=True) #add ticks left
ax.tick_params(axis='both', which='major', labelsize=20, width=3.5, length=20) #formatta ticks
plt.locator_params(axis='x', nbins=4) #presenta le labels
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23),fontsize=50,ncol=3,framealpha=0,columnspacing=0.5,handletextpad=0.4,prop=font) #formatta legenda
plt.yticks(fontsize=60,color='k')
plt.ylabel('Score', fontsize=60,color='k')
plt.yticks([1,20,40,60,80,100])
ax.set_ylim([1, 100])
plt.xticks(fontsize=60,color='k')
plt.xlabel('User', fontsize=60, color='k')
plt.margins(-0.07, 0) #riduci margini tra plot e bordo
fig = ax.get_figure()
plt.gcf().subplots_adjust(bottom=0.2) #add space down
plt.gcf().subplots_adjust(left=0.15) #add space left

#add small space between boxes of same group
from matplotlib.patches import PathPatch
def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
adjust_box_widths(fig, 0.90)

plt.savefig('Fig_c.pdf',bbox_inches='tight')
plt.close()
