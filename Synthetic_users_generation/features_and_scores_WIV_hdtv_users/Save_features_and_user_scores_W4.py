import csv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

c=0
user_score_hdtv=[]
experience_list_hdtv=[]
temp=[]

#collect experience name and each scores
with open('./original_waterlooIV_dataset/data.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
            row[2] = row[2].replace('[', '')
            row[2] = row[2].replace(']', '')
            a=row[2].split(' ')
            if row[6] == 'hdtv':  # and row[5]=='H264':
                temp = []
                for i in a:
                    temp.append(int(i))
                user_score_hdtv.append(temp)
                experience_list_hdtv.append(row[0])

#for each users the information about experienceid and his score usersxexperiences 31x450
user_experienceid_score_hdtv=[]
temp=[]

for i in range(len(experience_list_hdtv)):
    c = 1
    for k in user_score_hdtv[i]:
        user_experienceid_score_hdtv.append(['user_'+str(c)+'_hdtv',experience_list_hdtv[i],k])
        c+=1

# for each user collect the scores and experienceid
experienceid_score_hdtv_organized=[]
temp=[]
for k in range(1,33):
    for i in user_experienceid_score_hdtv:
        if i[0]=='user_'+str(k)+'_hdtv':
            temp.append([i[1],i[2]])
    experienceid_score_hdtv_organized.append(temp)
    temp=[]

# #plot scores by users_hdtv
# user_score=[]
# plt.figure('hdtv_scores_distribution')
# for i in range(0,32):
#     for k in experienceid_score_hdtv_organized[i]:
#         user_score.append(k[1])
#     #plt.plot(range(1,451),user_score,'o')
#     #plt.plot(user_score)
#     ecdf = sm.distributions.ECDF(user_score)
#     plt.step(ecdf.x, ecdf.y,label='user_'+str(i))
#     plt.xlabel('index of distorted experience',fontsize=18)
#     plt.ylabel('ECDF',fontsize=18)
#     plt.title('hdtv_scores_distribution',fontsize=18,fontweight="bold")
#     user_score=[]
# plt.legend(loc = 'upper left',ncol=4)

#collect the IFs for each experience
#hdtv
all_features=[]
for k in experienceid_score_hdtv_organized[0]:
    row_collect=[]
    with open('./original_waterlooIV_dataset/streaming_logs/' + str(k[0]), newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for row in csv_reader:

            for i in range(len(row)):
                row_collect.append(float(row[i]))
    #depend on what do you want
    all_features.append(row_collect)

#collect scores for each user of hdtv
users_score=[]
for i in range(0,32):
    user_score=[]
    for k in experienceid_score_hdtv_organized[i]:
        user_score.append(k[1])
    users_score.append(user_score)

np.save('./all_feat_hdtv.npy',all_features)
np.save('./users_scores_hdtv.npy',users_score)