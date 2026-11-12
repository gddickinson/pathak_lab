"""Defaults shared by the 2_analysis pipeline steps and the web app. Change a value here and every step follows.

PIXEL_SIZE_NM       Camera pixel size at the sample, in nm (108 nm: the lab's TIRF camera). Used wherever a step
                    converts between pixels and nm (Step 2 linking/saving tracks, Steps 3, 4 and 7). Set it to your
                    own camera's value; ThunderSTORM's x [nm]/y [nm] must have been exported with the same size.
FEATURE_TRANSFORM   Transform of the SVM features before PCA/scaling/SVC (Step 2 classification):
                    "boxcox" (default) Box-Cox, lambdas fitted on the training tracks and reused for prediction;
                    "none"   raw features: the legacy behaviour, which published results used (the old code meant to
                             Box-Cox transform but never did).
SKIP_FRAMES, LINK_DISTANCE, MIN_SEGMENTS   Step 2 linking settings (gap frames, max link distance in pixels,
                    tracks must have more than this many points).
NN_RADII            Step 8 neighbour-count radii (pixels).
"""

PIXEL_SIZE_NM = 108.0
FEATURE_TRANSFORMS = ("boxcox", "none")
FEATURE_TRANSFORM = "boxcox"
SKIP_FRAMES = 18
LINK_DISTANCE = 3
MIN_SEGMENTS = 6
NN_RADII = (3, 5, 10, 20, 30)
