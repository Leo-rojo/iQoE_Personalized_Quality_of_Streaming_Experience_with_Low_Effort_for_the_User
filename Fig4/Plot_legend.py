import pylab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
colori=cm.get_cmap('tab10').colors



# create a figure for the data
figData = pylab.figure()
ax = pylab.gca()
barWidth=3
a=np.arange(0,8,1)
b=[i+barWidth-0.05 for i in a]
c=[i+barWidth-0.05 for i in b]
d=[i+barWidth-0.05 for i in c]
e=[i+barWidth-0.05 for i in c]
f=[i+barWidth-0.05 for i in c]
g=[i+barWidth-0.05 for i in c]
h=[i+barWidth-0.05 for i in c]
l=[i+barWidth-0.05 for i in c]
m=[i+barWidth-0.05 for i in c]

plt.bar(a, [1,2,3,4,5,6,7,8], color='r', width=barWidth, linewidth=0, label='iQoE', align='edge')
plt.bar(b, [1,2,3,4,5,6,7,8], color=colori[1], width=barWidth, linewidth=0, label='B', align='edge')
plt.bar(c, [1,2,3,4,5,6,7,8], color=colori[2], width=barWidth, linewidth=0, label='L', align='edge')
plt.bar(d, [1,2,3,4,5,6,7,8], color=colori[8], width=barWidth, linewidth=0, label='V',align='edge')
plt.bar(e, [1,2,3,4,5,6,7,8], color=colori[7], width=barWidth, linewidth=0, label='P',align='edge')
plt.bar(f, [1,2,3,4,5,6,7,8], color=colori[6], width=barWidth, linewidth=0, label='S',align='edge')
plt.bar(g, [1,2,3,4,5,6,7,8], color=colori[9], width=barWidth, linewidth=0, label='F',align='edge')
plt.bar(h, [1,2,3,4,5,6,7,8], color=colori[4], width=barWidth, linewidth=0,label='A', align='edge')
plt.bar(l, [1,2,3,4,5,6,7,8], color='gold', width=barWidth, linewidth=0, label='N',align='edge')
plt.bar(m, [1,2,3,4,5,6,7,8], color=colori[0], width=barWidth, linewidth=0, label='Group-based iQoE', align='edge')
plt.legend(loc = 'upper left',ncol=10,frameon=False,fontsize=60,handletextpad=0.1)
# create a second figure for the legend
figLegend = pylab.figure(figsize = (20,10),dpi=100)

h,l=ax.get_legend_handles_labels()
# produce a legend for the objects in the other figure
pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',ncol=10,frameon=False,fontsize=100)
figLegend.savefig("legendgroup_v2.pdf",bbox_inches='tight')
plt.close()
plt.close()