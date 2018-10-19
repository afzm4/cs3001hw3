#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:55:28 2018

@author: root
"""

from id3 import Id3Estimator, export_text
import numpy as np

feature_names = ["home/away",
                 "top25",
                 "media"]

X = np.array([["home", "out", "1-NBC"],
              ["home", "in", "1-NBC"],
              ["away", "out", "2-ESPN"],
              ["away", "out", "3-FOX"],
              ["home", "out", "1-NBC"],
              ["away", "out", "4-ABC"]])
              
              

y = np.array(["(win)",
              "(lose)",
              "(win)",
              "(win)",
              "(win)",
              "(win)"])

clf = Id3Estimator()
clf.fit(X, y, check_input=True)

print(export_text(clf.tree_, feature_names))