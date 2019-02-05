from random import Random
from sklearn.decomposition.tests.test_truncated_svd import rng
from torch.utils.data import Dataset
class NaiveSequence():
    def __init__(self,symbolCount,length,seed=None):
        self.symbolCount=symbolCount
        if seed is not None:
            rng = Random(seed)
        else:
            rng=Random()
        self.length=length
        self.sequence=[rng.randrange(self.symbolCount) for _ in range(self.length) ]
    def get(self):
        return self.sequence
class NoiseFreeSymbol():
    def __init__(self,meanLength=5,maxLengthVariance=2,channels=2,rng=None):
        self.rng=rng
        self.channels=channels
        if self.rng==None:
            self.rng=Random()
        self.length=meanLength+self.rng.randint(-1*maxLengthVariance, maxLengthVariance)
        self.out=[self.rng.getrandbits(self.length) for _ in range(self.channels)]
    def get(self):
        return [('{0:'+'0'+str(self.length)+'b}').format(a) for a in self.out]
    def print(self):
        print(self.get())
    def getChannel(self,channel):
        return self.get()[channel]
    
class AudioSample():
    def __init__(self,phoneSequence,sequence):
        self.phoneSequence=phoneSequence
        self.sequence=sequence
    def print(self):
        for channel in range(self.phoneSequence[0].channels):
            channelout=""
            for phone in self.phoneSequence:
                channelout+=phone.getChannel(channel)
            print(channelout)
        print(self.sequence)
            
            
class ToyAudioSequenceDataset(Dataset):


    def __init__(self,phones,len,sequenceLengths=3,transform=None,seed=None,test=False,sequenceGenerator=NaiveSequence,sequenceComposer=None,signalIndependentNoiseGenerator=None):
        self.phones=phones
        self.len=len
        self.sequenceGenerator=sequenceGenerator
        self.transform=transform
        self.sequenceLengths=sequenceLengths
        if seed is None:
            self.rng=Random()
        else:
            self.rng=Random(seed)
        self.rngState=self.rng.getstate()
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        sequence=self.sequenceGenerator(len(self.phones),self.sequenceLengths,seed=idx).get()
        sample=AudioSample([self.phones[i] for i in sequence],sequence)
        if self.transform:
            sample= self.transform(sample)
        return sample

if __name__=="__main__":
    phone=NoiseFreeSymbol()
    phone.print()
    sequence=NaiveSequence(10,10)
    print(sequence.get())
    dataset=ToyAudioSequenceDataset([NoiseFreeSymbol(),NoiseFreeSymbol()],5)
    sample=dataset.__getitem__(0)
    sample.print()
    sample=dataset.__getitem__(1)
    sample.print()
