import pandas as pd
import numpy as np
from datetime import datetime

class SummaryStructure:
    def __init__(
        self, 
        min_fmics=5, 
        max_fmics=200, 
        threshold=1, 
        fuzziness=2):
        
        self.min_fmics = min_fmics
        self.max_fmics = max_fmics
        self.threshold = threshold
        self.fuzziness = fuzziness

        self.fmics = set()

        self.test = False
        self.purity = 0
        self.creations = 0
        self.removals = 0
        self.absorptions = 0
        self.merges = 0
    
    def clustering(self, datastream, test = False):
        self.test = test
        for example in datastream.itertuples():
            i_example = tuple([example.x, example.y])
            if len(self.fmics) < self.min_fmics:
                fmic = FMiC(i_example)
                self.fmics.add(fmic)
                if self.test: self.creations += 1
            else:
                all_dist = self.distances(i_example)
                is_outlier = True
                for i_fmic in self.fmics:
                    radius = 0
                    if i_fmic.N == 1:
                        radius = self.min_distance(i_fmic)
                    else:
                        radius = i_fmic.fuzzy_dispersion()
                    if all_dist[i_fmic] <= radius:
                        is_outlier = False
                        i_fmic.timestamp = datetime.today().timestamp()
                if is_outlier:
                    if len(self.fmics) == self.max_fmics:
                        self.remove_oldest()
                        if self.test: self.removals += 1
                    fmic = FMiC(i_example)
                    self.fmics.add(fmic)
                    if self.test: self.creations += 1
                else:
                    m_degrees = self.calc_m_degrees(all_dist)
                    self.update_fmics(i_example, all_dist, m_degrees)
                    if self.test: self.absorptions += 1
                self.merge_fmics()
        if self.test: self.evaluate_purity(datastream)
    
    def evaluate_purity(self, datastream):
        self.purity = 0       

    def euclidean_distance(self, a, b):
        return (((a[0]-b[0])**2 + (a[1]-b[1])**2) ** (1/2))

    def min_distance(self, i_fmic):
        res = 999999
        for j_fmic in self.fmics:
            if i_fmic == j_fmic: continue
            res = min(res, self.euclidean_distance(i_fmic.prototype(), j_fmic.prototype()))
        return res
    
    def distances(self, example):
        res = dict()
        for i_fmic in self.fmics:
            res[i_fmic] = self.euclidean_distance(example, i_fmic.prototype())
        return res

    def remove_oldest(self):
        min_timestamp = list(self.fmics)[0].timestamp
        for i_fmic in self.fmics:
            min_timestamp = min(min_timestamp, i_fmic.timestamp)
        for i_fmic in self.fmics:
            if i_fmic.timestamp == min_timestamp:
                self.fmics.remove(i_fmic)
                return

    def calc_m_degrees(self, all_dist):
        m_degrees = dict()
        for i_fmic in self.fmics:
            i_dist = all_dist[i_fmic]
            m_degree = 0
            for j_fmic in self.fmics:
                j_dist = all_dist[j_fmic]
                m_degree += (i_dist/j_dist) ** (2./(self.fuzziness-1))
            m_degrees[i_fmic] = 1. / m_degree
        return m_degrees

    def update_fmics(self, i_example, all_dist, m_degrees):
        for i_fmic in self.fmics:
            m_degree = m_degrees[i_fmic]
            dist = all_dist[i_fmic]
            #i_fmic.SSD += (m_degree**self.fuzziness) * (dist**2)
            i_fmic.SSD += m_degree * (dist**2)
            i_fmic.CF += np.array(i_example) * m_degree
            i_fmic.N += 1
            i_fmic.M += m_degree

    def merge_fmics(self):
        merge = False
        for i_fmic in self.fmics:
            if merge: break
            similarity = 0
            for j_fmic in self.fmics:
                if i_fmic == j_fmic: continue
                similarity = self.similarity(i_fmic, j_fmic)
                if similarity > self.threshold:
                    merge = True
                    i_fmic.N += j_fmic.N
                    i_fmic.M += j_fmic.M
                    i_fmic.SSD += j_fmic.SSD
                    i_fmic.CF += j_fmic.CF
                    break
        if merge:
            self.fmics.remove(j_fmic)
            if self.test: self.merges += 1

    def similarity(self, i_fmic, j_fmic):
        similarity = 0
        sum_radius = i_fmic.fuzzy_dispersion() + j_fmic.fuzzy_dispersion()
        dist = self.euclidean_distance(i_fmic.prototype(), j_fmic.prototype())
        similarity = sum_radius / dist
        return similarity

    def to_dataframe(self):
        data = {'x': [], 'y': [], 'radius': [], 'N': [], 'M': [], 'SSD': [], 'CF': []}
        for i_fmic in self.fmics:
            prototype = i_fmic.prototype()
            data['x'].append(prototype[0])
            data['y'].append(prototype[1])
            data['N'].append(i_fmic.N)
            data['M'].append(i_fmic.M)
            data['SSD'].append(i_fmic.SSD)
            data['CF'].append(i_fmic.CF)
            if i_fmic.N == 1:
                data['radius'].append(self.min_distance(i_fmic))
            else:
                data['radius'].append(i_fmic.fuzzy_dispersion())
        return pd.DataFrame(data)

class FMiC:
    def __init__(self, example):
        self.N = 1
        self.M = 1
        self.SSD = 0
        self.CF = example
        self.timestamp = datetime.today().timestamp()
    
    def fuzzy_dispersion(self):
        return (self.SSD / self.N) ** (1/2)

    def prototype(self):
        return np.array(self.CF) / self.M
