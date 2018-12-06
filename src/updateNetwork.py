__author__ = "Zafeirios Fountas, Kyriacos Nikiforou, Warrick Roseboom"
__credits__ = ["David Bhowmik", "Murray Shanahan", "Anil Seth"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Zafeirios Fountas"
__email__ = "fountas@outlook.com"
__status__ = "Published"

import numpy as np
import numpy.matlib as mat
import pickle

from src.plot_experiment import plotExperiment


def euc_dist(X,Y) :
    return np.sqrt(np.sum(np.square(X-Y)))

# net: Structure that contains the hierarchical neural network (MLP)
class UpdateNetwork :

    def __init__(self, params) :
        self.params = params

        # The array that keeps track of the accumulator - i.e. the salient features
        self.salientFeatures = {0 : [], 1 : [], 2 : [], 3 : []}

        # Self explanatory
        self.time = 0

	   # Initialise threshold to maximu
        self.thres = list(self.params['Tmax']['d2'])

        # Arrays that keep track of the pictures of the salient frames
        self.pictures = []
        # Arrays that keep track of the distances between the salient frames
        self.distances = {0 : [], 1 : [], 2 : [], 3 : [],\
                        'T0' : [], 'T1' : [], 'T2' : [], 'T3' : []}

        print("Starting experiment..")

        self.last = dict()
        self.prev = dict()
        self.last_t = dict()

        # Variable that accumulates how many salient features have been recorded.
        self.accumulator = {0:0, 1:0, 2:0, 3:0}

        # NORM FASEIS - Dave's stuff
        # Buffer of last distances to be normalized and used for the feat. threshold
        self.buff_dist = {0:[], 1:[], 2:[], 3:[]}

    # An iteration of the trial
    def run(self, image, features, output_prob, net_states, net_labels) :
        self.string  = " - Time:" + str(self.time)
        self.string += ", Feature:" + str(self.accumulator[3])

        # Calculate the distance between the activities of the two frames
        # in the same layer
        for i in range(4) :
            if self.time == 0 :
                self.dist = 0.0
                self.last_t[i] = 0
                self.last[i] = mat.zeros(len(features[i]))
                self.prev[i] = mat.zeros(len(features[i]))

            if self.params['type'] == 'prev' and self.time > 0 :
                self.dist = euc_dist(self.prev[i], features[i])
            elif self.params['type'] == 'last' and self.time > 0 :
                self.dist = euc_dist(self.last[i], features[i])

            # Calculate attention threshold

            self.D = np.abs(self.time - self.last_t[i])
            self.Tmin = self.params['Tmin']['d2'][i]
            self.Tmax = self.params['Tmax']['d2'][i]

            self.thres[i] = self.thres[i]
            self.thres[i] -= ((self.Tmax-self.Tmin)/self.params['Ttau'][i])*\
                             np.exp(-self.D/self.params['Ttau'][i])
            self.thres[i] += np.random.normal(0,(self.Tmax-self.Tmin)/50.0)

            # Reset feature accumulators
            if self.time == 0 or self.dist >= self.thres[i] :
                self.last_t[i] = self.time
                self.last[i] = features[i]
                self.thres[i] = self.Tmax
                self.accumulator[i] += 1

                # Features of the 3rd layer
                if i == 3:
                    self.string += "   NEW FEATURE NOW!!"
                    if self.params['visuals'] :
                        # Record new picture and distance
                        self.pictures.append(image)


            self.prev[i] = features[i]
            self.distances[i].append(self.dist)
            self.distances['T'+str(i)].append(self.thres[i])

            self.salientFeatures[i].append(self.accumulator[i])

        # MAKE PREDICTION
        self.now_xx = [self.accumulator[oo] for oo in self.accumulator.keys()]

        print(self.string, self.accumulator)

        self.time += 1


    def plot(self, net_states, net_labels, real_time, last_estimation):
        plotExperiment(self.time, self.salientFeatures,
                       last_estimation, self.pictures,
                       self.distances, self.params, True, net_states,
                       net_labels, real_time)
