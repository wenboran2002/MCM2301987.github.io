import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
class Data_used(object):
    def __init__(self,data):
        # number of reported results
        self.norr=np.array(data['Number of  reported results'])
        # percentage of hard
        self.poh=np.array(data['Number in hard mode']/self.norr)
        self.vowel=np.array(data['vowel'])
        self.repetition=np.array(data['repeatedletter'])
        self.frequency=np.array(data['functionoffrequency'])
        self.frequency_raw=np.array(data['frequency'])
        self.repe_score=np.array(data['score'])*10
        self.difficulty=np.array(data['difficulty'])
