import essentia
import numpy as np
import essentia.standard
import copy
import json
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

class SFA_HPCP_Extractor(object):
    def __init__(self,sample_time_epoch=0.1,word_length=4,threshold_ratio = 0.7,use_eqal_depth = False):

        self.use_eqal_depth = use_eqal_depth

        SpectralPeaks = essentia.standard.SpectralPeaks()
        Spectrum = essentia.standard.Spectrum()
        HPCP = essentia.standard.HPCP()

        self.extractor = lambda audio:HPCP(*SpectralPeaks(Spectrum(audio)))
        self.threshold_ratio = threshold_ratio

        self.word_length = word_length

        self.sample_time_epoch = sample_time_epoch
        self.sample_rate = 44100

        self.sample_window = int(sample_time_epoch * self.sample_rate)

    def Extract(self,audio,ratio = 1.0):

        if ratio < 1:
            audio = audio[len(audio)*(1-ratio):]

        length = len(audio) - self.sample_window - 1

        HPCPs = []

        for i in xrange(0,len(audio),self.sample_window):
            current_frame = audio[i:i+self.sample_window]
            if len(current_frame)%2 == 1:
                current_frame = current_frame[:-1]
            HPCPs.append(self.extractor(current_frame))

        HPCPs = np.asarray(HPCPs)

        if not self.use_eqal_depth:
            thresholds = np.asarray([self.threshold_ratio] * HPCPs.shape[1])
        else:
            if self.threshold_ratio == 0.5:
                thresholds = np.median(HPCPs,axis=0)
            else:
                thresholds = []
                transposed_HPCPs = HPCPs.transpose()
                for pitch in transposed_HPCPs:
                    threshold = np.sort(pitch)[int(self.threshold_ratio*len(pitch))]
                    thresholds.append(threshold)
                thresholds = np.asarray(thresholds)

        extracted_bit = (HPCPs > thresholds).astype(bool)

        #extracted_bit_diff = np.abs(extracted_bit[1:] - extracted_bit[:-1]).astype(bool)

        features = []

        candidates = np.asarray(range(12))

        time_offset = []

        def generate_next_bit(i,current_list,t):
            if len(current_list) == self.word_length:
                features.append(current_list)
                time_offset.append(t)
                return

            if i >= extracted_bit.shape[0]:
                return

            if len(current_list) > 0 and extracted_bit[i][current_list[-1]] == True:
                return generate_next_bit(i+1,current_list,t)

            for value in candidates[extracted_bit[i]]:
                generate_next_bit(i+1,current_list + [value],t)


        for i in range(length):
            generate_next_bit(i,[],i*self.sample_time_epoch)

        features = np.asarray(features)

        power_base = np.power(12,np.asarray(range(self.word_length))).astype(int)

        #print power_base

        extracted_sequence = np.matmul(features ,power_base )

        return zip(time_offset,extracted_sequence)

        # Generate characteristics

        characteristics = []
        current_queue = [extracted_sequence[0]]
        for i in xrange(1,len(extracted_sequence)):
            current_value = extracted_sequence[i]
            current_time_offset = (i - self.word_length + 1) * self.sample_time_epoch

            if current_queue[-1] != current_value:
                current_queue.append(current_value)

                if len(current_queue) > self.word_length:
                    del current_queue[0]

                if len(current_queue) == self.word_length:
                    characteristics.append((current_time_offset,copy.copy(current_queue)))

        return characteristics

    def ExtractFromFile(self,filename,ratio = 1.0):
        loader = essentia.standard.MonoLoader(filename=filename)
        audio = loader()
        return self.Extract(audio,ratio)

def Generate_one(path):
    #print path
    extractor = SFA_HPCP_Extractor()
    result = extractor.ExtractFromFile(filename=path)
    return (path,result)

def Generate_all():
    import os
    rootdir = '/Users/hedunbang/Music/mlib/'

    path = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path.append(os.path.join(subdir, file))

    pool = multiprocessing.Pool(8)

    dic = dict(pool.map(Generate_one,path))

    json.dump(dic,open("result.json",'w'))
    #print dic

if __name__ == '__main__':
    Generate_all()


