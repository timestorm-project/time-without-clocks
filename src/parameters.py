__author__ = "Zafeirios Fountas, Kyriacos Nikiforou, Warrick Roseboom"
__credits__ = ["David Bhowmik", "Murray Shanahan", "Anil Seth"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Zafeirios Fountas"
__email__ = "fountas@outlook.com"
__status__ = "Published"

class Parameters :
    def __init__(self) :
        # ------ EXPERIMENT PARAMETERS ----------------------------------------
        self.params = dict()

        # speed each object is presented for (max value for each = 64) in each trial
        # The original FPS of the video! NOTE: Needs to be a float!
        self.params['FPS'] = 30.0

        # ------ OTHER PARAMETERS ---------------------------------------------
        self.params['type'] = 'prev'                    # Distance is measured from either the 'last'
                                                        # or the 'prev' recorded feature

        self.params['xmax'] = 200.0                     # max graph xaxis goes up to

        self.params['type'] = 'prev'                    # Distance is measured from either the 'last'
                                                        # or the 'prev' recorded feature

        self.params['Tmin'] ={'d2':[100.0, 100.0, 5.0, 0.15]}         # min of threshold

        self.params['Tmax'] ={'d2':[340.0, 400.0, 35.0, 0.55]}         # max of threshold

        self.params['Ttau'] = [100.0, 100.0, 100.0, 100.0, 100.0]

        # ------ VISUALIZATION PARAMETERS --------------------------------------
        self.params['imageX'] = 92                   # The pixels of the image
        self.params['imageY'] = 92                   # -----------------------
        self.params['visuals'] = False               # Whether we'll use a visual interface
        self.params['xmax'] = self.params['xmax']    # max graph x axis goes up to
        self.params['rate'] = 20                     # How many frames visualization will skip every time!
        self.params['record_video'] = False          # Whether we plot and save every frame to produce a video
