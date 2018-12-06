__author__ = "Zafeirios Fountas, Kyriacos Nikiforou, Warrick Roseboom"
__credits__ = ["David Bhowmik", "Murray Shanahan", "Anil Seth"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Zafeirios Fountas"
__email__ = "fountas@outlook.com"
__status__ = "Published"

import sys, numpy as np, caffe, urllib.request, os.path

"""
data   - 	(10, 3, 227, 227) 	   length 154587
conv1  - 	(10, 96, 55, 55) 	   length 290400
norm1  - 	(10, 96, 55, 55) 	   length 290400
pool1  - 	(10, 96, 27, 27) 	   length 69984
conv2  - 	(10, 256, 27, 27) 	   length 186624
norm2  - 	(10, 256, 27, 27) 	   length 186624
pool2  - 	(10, 256, 13, 13) 	   length 43264
conv3  - 	(10, 384, 13, 13) 	   length 64896
conv4  - 	(10, 384, 13, 13) 	   length 64896
conv5  - 	(10, 256, 13, 13) 	   length 43264
pool5  - 	(10, 256, 6, 6) 	   length 9216
fc6    - 	(10, 4096)            length 4096
fc7    - 	(10, 4096)            length 4096
fc8    - 	(10, 1000)            length 1000
prob   - 	(10, 1000)            length 1000
"""


class AlexNet :
    BUFF_SIZE = 10000

    def __init__(self) :

        self.directory = 'alexnet'

        # Set the right path to your model definition file, pretrained model weights,
        # and the image you would like to classify.
        self.MODEL_DEF = self.directory + '/deploy.prototxt'
        self.MODEL_WEIGHTS = self.directory + '/bvlc_alexnet.caffemodel'

        if not os.path.isfile(self.MODEL_WEIGHTS):
            print('Downloading alexnet weights...')
            urllib.request.urlretrieve("http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel", self.MODEL_WEIGHTS)
            print('OK')

        # load the model
        if False: # GPU
            caffe.set_mode_gpu()
            caffe.set_device(0)
        self.net = caffe.Classifier(self.MODEL_DEF,     # Defines the structure of the model
                                    self.MODEL_WEIGHTS) # Contains the trained weights

        self.labels_file = 'alexnet/synset_words.txt'
        self.labels = np.loadtxt(self.labels_file, str, delimiter='\t')
        print("successfully loaded classifier")

        self.states = dict()
        for n in range(1000) :
            self.states[n] = []
        self.acc_states = dict()
        for n in range(1000) :
            self.acc_states[n] = [0.0]

        self.features = dict()

    def set_transformer_for_opencv_webcam(self) :
        # create transformer (configure preprocessing) for the input called 'data'
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))   # move image channels to outermost dimension

    def run(self, frame) :
        if frame == "" :
            frame = self.loaded_image
        self.transformed_image = self.transformer.preprocess('data', frame)
        # copy the image data into the memory allocated for the net
        self.net.blobs['data'].data[...] = self.transformed_image
        ### perform classification
        self.output = self.net.forward()
        # the output probability vector for the first image in the batch
        self.output_prob = self.output['prob'][0]
        new_label = str(round(max(self.output_prob),2))
        #if max(self.output_prob) > 0.3 :
        #    new_label += " - " + self.labels[self.output_prob.argmax()][10:]
        if max(self.output_prob) > 0.1 :
            new_label = self.labels[self.output_prob.argmax()].split(',')[0][10:]
        else:
            new_label = "-"
        for n in range(1000) :
            self.states[n].append(self.output_prob[n])
            self.acc_states[n].append(sum(self.states[n][-self.BUFF_SIZE:]))

        self.features[3] = np.array(self.output_prob)
        self.features[2] = np.array(self.net.blobs['fc7'].data[0])
        self.features[1] = np.array(self.net.blobs['pool5'].data[0].flatten())
        self.features[0] = np.array(self.net.blobs['conv2'].data[0][0].flatten())

        return new_label
