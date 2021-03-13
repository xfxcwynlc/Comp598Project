#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 03:05:29 2021

@author: jiuqiwang
"""

import numpy as np
import tarfile

with tarfile.open("./dataset/SHREC2017.tar.gz", "r:gz") as tar:
    print(tar.extract('train_gestures.txt'))

