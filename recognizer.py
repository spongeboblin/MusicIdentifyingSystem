import numpy as np
import warnings
warnings.filterwarnings("ignore")

class SimpleRecognizer(object):

    def __init__(self,database_dict,hpcp_count = 12,word_length = 4):
        songs = database_dict.keys()
        self.song_lengths = {}
        self.database = dict([(i,dict([(x,[]) for x in songs])) for i in range(hpcp_count**word_length)])

        for songname,features in database_dict.iteritems():
            features = np.asarray(features)
            self.song_lengths[songname] = np.max(features[:,0])*10+1
            for feature_value in np.unique(features[:,1]):
                selector = features[:,1] == feature_value
                time_offset_list = features[:,0][selector]
                self.database[feature_value][songname] = time_offset_list * 10

    # parameter to be tuned
    width = 2
    def maxCount (self, dots):
        ret = 0
        left = right = 0
        while (right < len(dots)):
            right = right + 1
            while (dots[right-1] - dots[left] > self.width):
                left = left + 1
            if (ret < right - left):
                ret = right -left
                #print ret, right, left
        return ret

    def recognize (self, filefea):
        matches = {}

        for name,length in self.song_lengths.iteritems():
            matches[name] = np.zeros(length)

        for feature in filefea :
            feature_value = feature[0]
            time_offset = feature[1]
            if feature_value in self.database:
                for songName,song_time_offsets in self.database[feature_value].iteritems() :
                    if len(song_time_offsets) <= 0:
                        continue
                    real_offsets = np.asarray(song_time_offsets) - time_offset * 10
                    real_offsets = real_offsets[real_offsets>=0].astype(int)
                    matches[songName][real_offsets] += 1

        score = {}
        for match,val in matches.items() :
            score[match] = np.convolve(val,np.asarray([1]*3)).max()
            continue
            #points = sorted(match[1])
            ## print "p", points
            #s = self.maxCount(points)
            #score.append( (match[0], s) )

        rank = sorted(score.iteritems(), key = lambda item: item[1],reverse=True)
        #print rank

        return rank[0][0]
