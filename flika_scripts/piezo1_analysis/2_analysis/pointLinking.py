"""Qt-free particle linking used by Step 2 (and the web interface).

PointLinker links localisations (rows of [frame, x, y]) into tracks frame by frame: each unassigned point starts a
track, which is extended to the nearest unassigned point in the next frame within `maxDistance` pixels, looking up to
`maxFramesSkipped` frames ahead. Tracks are built recursively; recursion stops at depth 10000 and sets
`recursiveFailure` (Step 2 then skips the file unless includeRecursionProblem is set).

Step 2's Points class subclasses PointLinker and adds the FLIKA display and image-intensity methods.
"""
import sys

import numpy as np

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
