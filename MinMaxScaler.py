# -*- coding: utf-8 -*-

import numpy as np

class MinMaxScaler(object):
    def __init__(self,data=None,feature_range=(-1,1)):
        
        self.fit(data, feature_range)
        
    
    def fit(self,data,feature_range=(-1,1)):
        
        self.Max = np.max(data)
        self.Min = np.min(data)
        self.feature_range = feature_range
        
    
    def scale(self, data):
    
        assert self.Max is not None
        scaled = data
        scaled = (data - self.Min) * (self.feature_range[1] - self.feature_range[0]) / (self.Max - self.Min) + self.feature_range[0]
    
        return scaled
    
    def scale_inverse(self, scaled):

        assert self.Max is not None
        data = scaled
        data = (scaled - self.feature_range[0]) * (self.Max - self.Min) / (self.feature_range[1] - self.feature_range[0]) + self.Min

        return data