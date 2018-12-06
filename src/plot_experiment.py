__author__ = "Zafeirios Fountas, Kyriacos Nikiforou, Warrick Roseboom"
__credits__ = ["David Bhowmik", "Murray Shanahan", "Anil Seth"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Zafeirios Fountas"
__email__ = "fountas@outlook.com"
__status__ = "Published"

import numpy as np
from matplotlib import pyplot as plt
import os

def plotExperiment(t, salientFeatures, last_estimation, pictures,
                   distances, params, save_it, states, labels, real_time) :

    if t == 0 or not real_time:
        plt.figure(figsize=(15,10))
        if real_time :
            plt.ion()
        else :
            plt.ioff()

    real_time_now = t/params['FPS']

    if real_time :
        plt.clf()

    plt.subplot(231)
    for key in states.keys() :
        if len(states[key]) and max(states[key]) > 0.3 :
            p = plt.plot(states[key])
            if max(states[key]) > 0.3 :
                 plt.annotate(labels[int(key)][10:], xy=(2, 1),
                             xytext=(np.argmax(states[key]), np.max(states[key])),
                             color=p[0].get_color())
    plt.ylim(0, 1)



    plt.subplot(232)
    if pictures != [] :
        plt.imshow(pictures[-1])
        if len(salientFeatures[3]) > 0 :
            plt.title("Feature:" + str(salientFeatures[3][-1]))


    for i in range(len(pictures)) :
        if i < 24 :
            plt.subplot(9, 9, 7 + i%3 + 9*(i/3))
            plt.imshow(pictures[i])
            plt.xticks([])
            plt.yticks([])



    for i in range(4) :
        plt.subplot(8,3,13+3*(3-i))
        if len(salientFeatures[i]) > 0 :
            plt.plot(salientFeatures[i],lw=2.0)
            plt.ylabel('Layer '+str(i))
            if max(salientFeatures[i]) > 0.0 :
                plt.ylim(0, max(salientFeatures[i])*1.1)
        if i == 0 :
            plt.xlabel('time')
        else :
            plt.xticks([])
    plt.title('Salient Features')

    for i in range(4) :
        plt.subplot(8,3,14+3*(3-i))
        if len(distances[i]) > 0 :
            plt.plot(distances[i])
            plt.plot(distances['T'+str(i)],c='g',lw=2.0)
            plt.ylabel('layer '+str(i))
            #plt.ylim(0, 1.3)
        if i == 0 :
            plt.xlabel('time')
        else :
            plt.xticks([])
    plt.title('Euclidean distance (current-'+params['type']+')')

    plt.subplot(8,3,24)

    # Plot real time
    plt.plot([real_time_now, real_time_now], [0,1], c='r', lw=2.0,label='Real')
    plt.plot([last_estimation, last_estimation], [0,1], c='b', lw=2.0,label='Estimation')
    plt.xlim(0, max(real_time_now, last_estimation)*1.5)
    plt.xlabel('time (sec)')


    if save_it :
        if not os.path.isdir("output") :
            os.makedirs("output")
        plt.savefig("output/Episode-"+str(t).zfill(4)+'-'+params['type']+".png")

    if params['record_video'] :
        plt.savefig(params['folder'] +'Frame'+str(t).zfill(4)+'.png')

    if real_time :
        plt.pause(0.001)#0.05)
    else :
        plt.show()
