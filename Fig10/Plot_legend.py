import pylab
import matplotlib.pyplot as plt
import numpy as np

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
plt.bar(a, [1,2,3,4,5,6,7,8], color ='w', width = barWidth-0.1,linewidth=0,label='Testing ABR:',align='edge')
plt.bar(b, [1,2,3,4,5,6,7,8], color ='r', width = barWidth-0.1,linewidth=0,label='BBA',align='edge')
plt.bar(c, [1,2,3,4,5,6,7,8], color ='g', width = barWidth-0.1,linewidth=0,label='TR',align='edge')
plt.bar(d, [1,2,3,4,5,6,7,8], color ='b', width = barWidth-0.1,linewidth=0,label='MPC',align='edge')

# create a second figure for the legend
figLegend = pylab.figure(figsize = (20,10),dpi=100)

# produce a legend for the objects in the other figure
pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left',ncol=5,frameon=False)
figLegend.savefig("legendabr.pdf",bbox_inches='tight')
plt.close()
plt.close()