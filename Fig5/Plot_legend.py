import pylab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
colori=cm.get_cmap('tab10').colors
font_general = {'family' : 'sans-serif',
                        #'weight' : 'bold',
                        'size'   : 50}
plt.rc('font', **font_general)

# create a figure for the data
figData = pylab.figure()
ax = pylab.gca()

a=np.arange(0,8,1)
barWidth=3
b=[i+barWidth-0.05 for i in a]
c=[i+barWidth-0.05 for i in b]
d=[i+barWidth-0.05 for i in c]
plt.bar(b, [1,2,3,4,5,6,7,8], color ='r', width = barWidth-0.1,linewidth=0,label='iQoE',align='edge')
plt.bar(c, [1,2,3,4,5,6,7,8], color =colori[8], width = barWidth-0.1,linewidth=0,label='V',align='edge')
plt.bar(d, [1,2,3,4,5,6,7,8], color =colori[4], width = barWidth-0.1,linewidth=0,label='A',align='edge')

# create a second figure for the legend
figLegend = pylab.figure(figsize = (20,10),dpi=100)

# produce a legend for the objects in the other figure
pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',ncol=5,frameon=False)
figLegend.savefig("legendmrg.pdf",bbox_inches='tight')
plt.close()
plt.close()