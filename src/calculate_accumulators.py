__author__ = "Zafeirios Fountas, Kyriacos Nikiforou, Warrick Roseboom"
__credits__ = ["David Bhowmik", "Murray Shanahan", "Anil Seth"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Zafeirios Fountas"
__email__ = "fountas@outlook.com"
__status__ = "Published"

import numpy as np

"""
This class reads a timeseries of distances between snapshots of alexnet activation
(one timeseries for each layer for each trial) and calculates accumulators for
each layer.
"""
class Accummulators :

    # ------ ATTENTION PARAMETERS ---------------------------------------------
    params = dict()
    params['Tmin'] ={'d2':[100.0, 100.0, 5.0, 0.15]}         # min of threshold
    params['Tmax'] ={'d2':[340.0, 400.0, 35.0, 0.55]}         # max of threshold
    params['Ttau'] = [100.0, 100.0, 100.0, 100.0]
    params['Trand'] = 50.0
    params['Trand_m'] = 0.0


    def calculate(self, distances = [[], [], [], []], extra_dp_freq = 0.0) :

        accumulator = [0.0, 0.0, 0.0, 0.0]
        last_t = {0 : 0.0, 1 : 0.0, 2 : 0.0, 3 : 0.0}
        thres = list(self.params['Tmax']['d2'])
        plot_y = {0 : [], 1 : [], 2 : [], 3 : []}
        extra_dp = {'x' : [], 'y' : []}

        for time in range(len(distances[0])) :
            for layer in range(4) :
                dist = distances[layer][time]

                # Calculate attention threshold
                D = np.abs(time - last_t[layer])
                Tmin = self.params['Tmin']['d2'][layer]
                Tmax = self.params['Tmax']['d2'][layer]
                thres[layer] = thres[layer] - ((Tmax-Tmin)/self.params['Ttau'][layer])*np.exp(-D/self.params['Ttau'][layer]) + np.random.normal(self.params['Trand_m'],(Tmax-Tmin)/self.params['Trand'])

                # Reset feature accumulators
                if  dist >= thres[layer] :
                    last_t[layer] = time
                    thres[layer] = Tmax
                    accumulator[layer] += 1

                plot_y[layer].append(thres[layer])

            # Add extra data points from the middle of the trial if requested
            if np.random.rand() < extra_dp_freq :
                extra_dp['x'].append([accumulator[i] for i in range(len(accumulator))])
                extra_dp['y'].append(time/30.0)

        return accumulator, plot_y, extra_dp
