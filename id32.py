#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 23:41:58 2018

@author: root
"""

from id3 import Id3Estimator, export_text
import numpy as np

feature_names = ["outlook",
                 "temperature",
                 "humidity",
                 "windy"]

X = np.array([["sunny", "hot", "high", "false"],
              ["sunny", "hot", "high", "true"],
              ["overcast", "hot", "high", "false"],
              ["rainy", "mild", "high", "false"],
              ["rainy", "cool", "normal", "false"],
              ["rainy", "cool", "normal", "true"],
              ["overcast", "cool", "normal", "true"],
              ["sunny", "mild", "high", "false"],
              ["sunny", "cool", "normal", "false"],
              ["rainy", "mild", "normal", "false"],
              ["sunny", "mild", "normal", "true"],
              ["overcast", "mild", "high", "true"],
              ["overcast", "hot", "normal", "false"],
              ["rainy", "mild", "high", "true"]])
              
              

y = np.array(["(no)",
              "(no)",
              "(yes)",
              "(yes)",
              "(yes)",
              "(no)",
              "(yes)",
              "(no)",
              "(yes)",
              "(yes)",
              "(yes)",
              "(yes)",
              "(yes)",
              "(no)"])

clf = Id3Estimator()
clf.fit(X, y, check_input=True)

print(export_text(clf.tree_, feature_names))