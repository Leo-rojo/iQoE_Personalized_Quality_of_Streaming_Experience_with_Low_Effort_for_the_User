import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
from matplotlib.patches import PathPatch
#need to be in folder a

#remove type3 font
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#collect mos and individual scores
xls = pd.ExcelFile(r"../hdtv_scores.xlsx") #use r before absolute file path
sheetX = xls.parse(0)
mosarray=sheetX['mos'].tolist()
collect_all=[]
users_scores=np.load('../users_scores_hdtv.npy')

#sorted users
right_order=[26	,8	,22	,31	,12	,16	,28	,11	,7	,30	,24	,10	,21	,5	,13	,15	,18	,2	,4	,3	,9	,17	,14	,23	,25	,20	,27	,19	,6	,32	,1	,29]

#collect users score plus mos scores in one array
ordered_users=[]
for num in right_order:
    ordered_users= ordered_users + users_scores[num - 1].tolist()
ordered_users= ordered_users + mosarray

#save users name for dataframe structure
users=[]
for u in right_order:
    for i in range(450):
        users.append('H_'+str(u))
for i in range(450):
    users.append('HA')

#create datafram for boxplot
df = pd.DataFrame({'Users':users,'ordered':ordered_users})
df = df[['Users','ordered']]
dd=pd.melt(df,id_vars=['Users'],value_vars=['ordered'],var_name='')
my_colors = ["#ffd700"]

#plot boxplot
fig = plt.figure(figsize=(20, 10),dpi=100)
font = font_manager.FontProperties(size=50)
sns.set(rc={'figure.figsize':(8,3)},style="white")
sns.set_palette(my_colors)
ax=sns.boxplot(x='Users',y='value',data=dd,hue='')#,palette=my_pal
ax.tick_params(bottom=True)
ax.tick_params(left=True)
ax.tick_params(axis='both',which='major', labelsize=20, width=3.5, length=20)
plt.locator_params(axis='x', nbins=5)
ax.set(ylabel='QoE')
xlabels=['H1'	,''	,''	,''	,''	,''	,''	,'H8'	,''	,''	,''	,''	,''	,''	,''	,'H16'	,''	,''	,''	,''	,''	,''	,''	,'H24'	,''	,''	,''	,''	,''	,''	,''	,'H32','']
ax.set_xticklabels(xlabels,fontsize=60,color='k')
ax.set_xticks([0,7,15,23,31])
plt.yticks(fontsize=60, color='k')
plt.ylabel('SA score', fontsize=60,color='k')
plt.xlabel('User', fontsize=60, color='k')
plt.yticks([1, 20, 40, 60, 80, 100])
fig = ax.get_figure()
gold_patch = mpatches.Patch(color='#ffd700', label='Users')
green_patch = mpatches.Patch(color='#4b8521', label='HA, average user')
plt.legend(handles=[gold_patch,green_patch],loc='upper center', bbox_to_anchor=(0.5, 1.23), fontsize=50, ncol=2, framealpha=0, columnspacing=0.5,handletextpad=0.4,prop=font)
plt.margins(0,0)
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)

#add small space between boxes of same group
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
adjust_box_widths(fig, 0.8)
plt.savefig('Fig_a.png',bbox_inches='tight')
