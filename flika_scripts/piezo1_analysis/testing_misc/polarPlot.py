#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 11:26:25 2023

@author: george
"""

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *


plot = pg.plot()
plot.setAspectLocked()

# Add polar grid lines
plot.addLine(x=0, pen=1)
plot.addLine(y=0, pen=1)
plot.getViewBox().invertY(True)
for r in range(5, 21, 5):
    r = r/10
    circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
    circle.setPen(pg.mkPen('w', width=0.8))
    plot.addItem(circle)

# make polar data
#theta = np.linspace(0, 2 * np.pi, 100)
#radius = np.random.normal(loc=10, size=100)

theta = np.radians(np.array([90]))
radius = np.array([0.5])


# Transform to cartesian and plot
x = radius * np.cos(theta)
y = radius * np.sin(theta)


for i in range(len(x)):

    path = QPainterPath(QPointF(0,0))
    path.lineTo(QPointF(x[i],y[i]))    

    item = pg.QtGui.QGraphicsPathItem(path)
    item.setPen(pg.mkPen('r', width=5))                
    plot.addItem(item)    

#position label
labels = [0,90,180,270]
d = 2
pos = [ (d,0),(0,d),(-d,0),(0,-d) ]
for i,label in enumerate(labels):
    text = pg.TextItem(str(label), color=(200,200,0))
    plot.addItem(text)
    text.setPos(pos[i][0],pos[i][1])

# scale
for r in range(5, 20, 5):
    r = r/10
    text = pg.TextItem(str(r))
    plot.addItem(text)
    text.setPos(0,r)




if __name__ == "__main__":
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()