__author__ = "Zafeirios Fountas, Kyriacos Nikiforou, Warrick Roseboom"
__credits__ = ["David Bhowmik", "Murray Shanahan", "Anil Seth"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Zafeirios Fountas"
__email__ = "fountas@outlook.com"
__status__ = "Published"

import pickle, os, cv2, time, numpy as np, matplotlib.pyplot as plt

from sklearn.svm import SVR
from sys import argv

from src.parameters import Parameters
from src.AlexNet import AlexNet
from src.updateNetwork import UpdateNetwork
from src.calculate_accumulators import Accummulators

frame_to_write = 0
moving_window_size = 5

if len(argv) == 3 and argv[1] == '-i':
    DEVICE = -1
    frame_path = []
    for file in os.listdir(argv[2]):
        if file.endswith(".jpg"):
            frame_path.append(os.path.join(argv[2], file))

    frame_path.sort()

    import re
    def natural_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
        return sorted(l, key = alphanum_key)
    frame_path = natural_sort(frame_path)
else:
    DEVICE = 0

params = Parameters().params
params['visuals'] = True
alexnet = AlexNet()
upNet = UpdateNetwork(params)
ac = Accummulators()
show_details = True

RESOLUTION = 800
if DEVICE >= 0:
    video_capture = cv2.VideoCapture(DEVICE)
font = cv2.FONT_HERSHEY_SIMPLEX
alexnet.set_transformer_for_opencv_webcam()
t = 0

all_x = []
all_y = []

ac = Accummulators()
start_time = time.time()
while DEVICE >= 0 or len(frame_path) > 0:
    # Capture frame-by-frame
    if DEVICE >= 0:
        ret, frame = video_capture.read()
    else:
        frame = cv2.imread(frame_path.pop(0))

    if np.shape(frame)[0] > np.shape(frame)[1]:
        onset = np.shape(frame)[0] - np.shape(frame)[1]
        frame = frame[int(onset/2):-int(onset/2),:,:]
    elif np.shape(frame)[0] < np.shape(frame)[1]:
        onset = np.shape(frame)[1] - np.shape(frame)[0]
        frame = frame[:,int(onset/2):-int(onset/2),:]

    frame_to_show = np.copy(frame)

    label = alexnet.run(frame)
    upNet.run(frame, alexnet.features, alexnet.output_prob, alexnet.states, alexnet.labels)
    distance = upNet.distances
    x, thresh_, extra_dp = ac.calculate(distance, extra_dp_freq=0)
    all_x.append(x)
    all_y.append(time.time()-start_time)

    to_show = cv2.resize(frame_to_show, (RESOLUTION,RESOLUTION))

    cv2.putText(to_show, "Training...", (20,100), font, 1.5, (255,255,255), 2)
    cv2.putText(to_show, "(when ready press 'Q')", (20,150), font, 1.0, (255,255,255), 2)

    cv2.imshow('img', to_show)
    cv2.imwrite('video/frame'+str(frame_to_write).zfill(5)+'.png', to_show)
    frame_to_write += 1
    print(time.time()-start_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


train_x = all_x
train_y = all_y


# Fit regression model
regressor = SVR(kernel='rbf', C=1e3, gamma=0.0001)

regressor.fit(train_x, train_y)


all_estimates = []
my_estimate = 0
t=0
frames = 0
FPS = []
passed_int = 0.0
start_time = time.time()
alexnet = AlexNet()
alexnet.set_transformer_for_opencv_webcam()
upNet = UpdateNetwork(params)
ac = Accummulators()

while DEVICE >= 0 or len(frame_path) > 0:
    # Capture frame-by-frame
    if DEVICE >= 0:
        ret, frame = video_capture.read()
    else:
        frame = cv2.imread(frame_path.pop(0))


    if np.shape(frame)[0] > np.shape(frame)[1]:
        onset = np.shape(frame)[0] - np.shape(frame)[1]
        frame = frame[int(onset/2):-int(onset/2),:,:]
    elif np.shape(frame)[0] < np.shape(frame)[1]:
        onset = np.shape(frame)[1] - np.shape(frame)[0]
        frame = frame[:,int(onset/2):-int(onset/2),:]

    frame_to_show = np.copy(frame)

    frames += 1
    passed = time.time()-start_time
    if passed_int-passed < -1.0 :
        passed_int += 1.0
        if len(FPS) > 10 : FPS = FPS[1:]
        FPS.append(frames)
        frames = 0

    label = alexnet.run(frame)
    upNet.run(frame, alexnet.features, alexnet.output_prob, alexnet.states, alexnet.labels)
    distance = upNet.distances
    x, thresh_, extra_dp = ac.calculate(distance, extra_dp_freq=0)


    # Estimation
    svr_rbf = regressor

    estimate = svr_rbf.predict(np.reshape(np.array(x),(1,4)))
    all_estimates.append(estimate)
    #print("Estimation:", estimate)

    if len(all_estimates) >= moving_window_size:
        all_estimates = all_estimates[1:]
    my_estimate = np.mean(all_estimates)

    t += 1

    params['FPS'] = np.mean(FPS)

    if show_details :
        details = 'FPS: '+str(round(params['FPS'],0))
        cv2.putText(frame_to_show, details, (20,340), font, 0.5,(255,255,255),1)
        cv2.putText(frame_to_show, label, (20,300), font, 1.1,(255,255,255),2)
        AA = 'Time: ' + str(round(passed,1))
        AA += ' estim: ' + str(round(my_estimate,2))
        cv2.putText(frame_to_show, AA, (20,270), font, 0.5, (255,255,255), 2)


    # Display the resulting frame
    to_show = cv2.resize(frame_to_show, (RESOLUTION,RESOLUTION))
    cv2.imshow('img', to_show)
    cv2.imwrite('video/frame'+str(frame_to_write).zfill(5)+'.png', to_show)
    frame_to_write += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
if DEVICE >= 0:
    video_capture.release()
cv2.destroyAllWindows()


upNet.plot(alexnet.states, alexnet.labels, False, all_estimates[-1])
