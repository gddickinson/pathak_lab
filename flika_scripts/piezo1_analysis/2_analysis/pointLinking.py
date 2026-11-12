"""Qt-free particle linking used by Step 2 (and the web interface).

PointLinker links localisations (rows of [frame, x, y]) into tracks frame by frame: each unassigned point starts a
track, which is extended to the nearest unassigned point in the next frame within `maxDistance` pixels, looking up to
`maxFramesSkipped` frames ahead. Tracks are built recursively; recursion stops at depth 10000 and sets
`recursiveFailure` (Step 2 then skips the file unless includeRecursionProblem is set).

getIntensities reads each point's mean 3x3-pixel intensity from the recording. Step 2's Points class subclasses
PointLinker and adds the FLIKA track display.
"""
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Deep recursion in extend_track (one level per linked point), as set in Step 2.
recursionLimit = 1000000
sys.setrecursionlimit(recursionLimit)


class PointLinker(object):
    """Link [frame, x, y] points into tracks (nearest neighbour, gap closing). No Qt or FLIKA needed."""
    def __init__(self, txy_pts):
        self.frames = np.unique(txy_pts[:, 0]).astype(int)
        self.txy_pts = txy_pts
        self.window = None
        self.pathitems = []
        self.pts_by_frame = []
        self.pts_remaining = []
        self.pts_idx_by_frame = []  # this array has the same structure as points_by_array but contains the index of the original txy_pts argument
        self.intensities = [] #GD edit
        self.recursiveFailure = False

    def link_pts(self, maxFramesSkipped, maxDistance):
        print('Linking points')
        self.pts_by_frame = []
        self.pts_remaining = []
        self.pts_idx_by_frame = []  # this array has the same structure as points_by_array but contains the index of the original txy_pts argument
        for frame in np.arange(0, np.max(self.frames) + 1):
            indicies = np.where(self.txy_pts[:, 0] == frame)[0]
            pos = self.txy_pts[indicies, 1:]
            self.pts_by_frame.append(pos)
            self.pts_remaining.append(np.ones(pos.shape[0], dtype=bool))
            self.pts_idx_by_frame.append(indicies)

        tracks = []
        for frame in self.frames:
            for pt_idx in np.where(self.pts_remaining[frame])[0]:
                self.pts_remaining[frame][pt_idx] = False
                abs_pt_idx = self.pts_idx_by_frame[frame][pt_idx]
                track = [abs_pt_idx]
                track = self.extend_track(track, maxFramesSkipped, maxDistance)
                tracks.append(track)
        self.tracks = tracks

    def extend_track(self, track, maxFramesSkipped, maxDistance, i=0):
        #this limits the amount of possible recursion - 10000 set by trial and error
        if i >= 10000:
            self.recursiveFailure = True
            return track

        pt = self.txy_pts[track[-1]]
        # pt can move less than two pixels in one frame, two frames can be skipped
        for dt in np.arange(1, maxFramesSkipped+2):
            frame = int(pt[0]) + dt
            if frame >= len(self.pts_remaining):
                return track
            candidates = self.pts_remaining[frame]
            nCandidates = np.count_nonzero(candidates)
            if nCandidates == 0:
                continue
            else:
                distances = np.sqrt(np.sum((self.pts_by_frame[frame][candidates] - pt[1:]) ** 2, 1))
            if any(distances < maxDistance):
                next_pt_idx = np.where(candidates)[0][np.argmin(distances)]
                abs_next_pt_idx = self.pts_idx_by_frame[frame][next_pt_idx]
                track.append(abs_next_pt_idx)
                self.pts_remaining[frame][next_pt_idx] = False
                track = self.extend_track(track, maxFramesSkipped, maxDistance, i=i+1)
                return track
        return track

    def get_tracks_by_frame(self):
        tracks_by_frame = [[] for frame in np.arange(np.max(self.frames)+1)]
        for i, track in enumerate(self.tracks):
            frames = self.txy_pts[track][:,0].astype(int)
            for frame in frames:
                tracks_by_frame[frame].append(i)
        self.tracks_by_frame = tracks_by_frame

    def getIntensities(self, dataArray):
        #intensities retrieved from image stack using point data (converted from floats to ints)

        n, w, h = dataArray.shape

        #clear intensity list
        self.intensities = []

        for point in tqdm(self.txy_pts):
            frame = int(round(point[0]))
            x = int(round(point[1]))
            y = int(round(point[2]))

            #set x,y bounds for 3x3 pixel square
            xMin = x - 1
            xMax = x + 2

            yMin = y - 1
            yMax = y + 2

            #deal with edge cases
            if xMin < 0:
                xMin = 0
            if xMax > w:
                xMax = w

            if yMin <0:
                yMin = 0
            if yMax > h:
                yMax = h

            #get mean pixels values for 3x3 square
            #self.intensities.append((np.mean(dataArray[frame][xMin:xMax,yMin:yMax]) - np.min(dataArray[frame]))) #no longer subtracting min
            self.intensities.append(np.mean(dataArray[frame][xMin:xMax,yMin:yMax]))


def addID(file):
    """Add ID column to CSV file"""
    df = pd.read_csv(file)
    df['id'] = df['id'].astype('int')
    df['frame'] = df['frame'].astype('int') -1 #reseting first frame to zero to maych flika display
    df.to_csv(file.split('.csv')[0] + 'ID.csv')


def load_points(filename, pixelsize = 108):
    """Load points from CSV file"""
    #filename = open_file_gui("Open points from thunderstorm csv, filetypes='*.csv")
    if filename is None:
        return None

    pointsDF = pd.read_csv(filename)
    pointsDF['frame'] = pointsDF['frame'].astype(int)
    pointsDF['x'] = pointsDF['x [nm]'] / pixelsize
    pointsDF['y'] = pointsDF['y [nm]'] / pixelsize
    pointsDF = pointsDF[['frame','x','y']]
    all_pts = pointsDF.to_numpy()

    print('--- Point Data loaded ---')
    print(pointsDF)

    return all_pts


def flatten(l):
    """Flatten a list of lists"""
    return [item for sublist in l for item in sublist]


def savetracksCSV(points, filename, locsFileName, noLocsFile=False):
    """Save tracks to CSV file"""
    tracks = points.tracks
    if isinstance(tracks[0][0], int):
        tracks = [[np.ndarray.item(a) for a in b] for b in tracks]
    txy_pts = points.txy_pts.tolist()

    txy_intensities = points.intensities

    txy_indexes = flatten(points.pts_idx_by_frame)


    #filter tracks list to only include linked tracks
    linkedTracks = [item for item in tracks if len(item) > 1]


    #get xy coordinates and intensities for linked tracks
    trackNumber = []

    txy_intensitiesByTrack = []

    txy_indexesByTrack = []


    # for i, indices in enumerate(linkedTracks):
    #     trackNumber.append( i )
    #     txy_ptsByTrack.append(list(txy_pts[j] for j in indices))
    #     #txy_intensitiesByTrack.append(list(txy_intensities[k] for k in indices))


    frameList = []
    xList = []
    yList = []

    for i, indices in enumerate(linkedTracks):
        ptsList = list(txy_pts[j] for j in indices)

        intensitiesList =list(txy_intensities[k] for k in indices)

        if noLocsFile == False:
            indexList =list(txy_indexes[l] for l in indices)

        for pts in ptsList:
            trackNumber.append(i)
            frameList.append(pts[0])
            xList.append(pts[1])
            yList.append(pts[2])


        for intensity in intensitiesList:
            txy_intensitiesByTrack.append(intensity)

        if noLocsFile == False:
            for ind in indexList:
                txy_indexesByTrack.append(ind)


    #make dataframe of tracks, xy coordianates and intensities for linked tracks
    if noLocsFile == False:
        dict = {'track_number': trackNumber, 'frame':frameList, 'x': xList, 'y':yList, 'intensity': txy_intensitiesByTrack, 'file_id': txy_indexesByTrack}
    else:
        dict = {'track_number': trackNumber, 'frame':frameList, 'x': xList, 'y':yList, 'intensity': txy_intensitiesByTrack}

    linkedtrack_DF = pd.DataFrame(dict)

    if noLocsFile == False:
        #match id to locs file values (starting at 1)

        #linkedtrack_DF['file_id'] = linkedtrack_DF['file_id'] + 1
        locs_DF = pd.read_csv(locsFileName)
        id_list = locs_DF['id'].tolist()
        file_id_list = linkedtrack_DF['file_id'].tolist()

        idsForlinkedPoints = [id_list[i] for i in file_id_list]

        linkedtrack_DF['id'] = idsForlinkedPoints
    else:
        linkedtrack_DF['id'] = linkedtrack_DF.index

    #convert back to nm
    linkedtrack_DF['x [nm]'] = linkedtrack_DF['x'] * 108
    linkedtrack_DF['y [nm]'] = linkedtrack_DF['y'] * 108

    #cast frames as int
    linkedtrack_DF['frame'] = linkedtrack_DF['frame'].astype('int')

    #round intensity
    linkedtrack_DF['intensity'] = linkedtrack_DF['intensity'].round(2)

    #drop file-id
    if noLocsFile == False:
        linkedtrack_DF = linkedtrack_DF.drop(['file_id'], axis=1)

    #save df as csv
    #linkedtrack_DF = linkedtrack_DF.sort_values('id')
    linkedtrack_DF.to_csv(filename, index=True)


    print('tracks file {} saved'.format(filename))
