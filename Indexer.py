# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:33:56 2015

@author: Stuart Grieve
"""

def get_index_of_min(Data_List):
    """
    Return as list of the indexes of the minmum values in a 1D array of data
    
    SWDG May 2015
    """
    import numpy as np
    
    #make sure data is in a standard list, not a numpy array
    if (type(Data_List).__module__ == np.__name__):
        Data_List = list(Data_List)
    
    #return a list of the indexes of the minimum values. Important if there is >1 minimum
    return [i for i,x in enumerate(Data_List) if x == min(Data_List)]


def get_index_of_max(Data_List):
    """
    Return as list of the indexes of the maximum values in a 1D array of data
    
    SWDG May 2015
    """
    import numpy as np    
    
    #make sure data is in a standard list, not a numpy array
    if (type(Data_List).__module__ == np.__name__):
        Data_List = list(Data_List)
    
    #return a list of the indexes of the max values. Important if there is >1 maximum
    return [i for i,x in enumerate(Data_List) if x == max(Data_List)]