from toydataset import ToyAudioSequenceDataset, NoiseFreeSymbol
from random import Random
import os
numphones=8
rng=Random(4)
phones=[NoiseFreeSymbol(channels=1,rng=rng,meanLength=64,maxLengthVariance=8) for _ in range(numphones)]

sequenceLengths=6
def writeDataset(dataset,file):
    with open("data"+os.path.sep+file,'w') as f:
        with open("data"+os.path.sep+file+".symbols",'w') as g:
            for i in range(len(dataset)):
                item=dataset.__getitem__(i)
                phoneStrings="".join(list(map(lambda a:a.get()[0],item.phoneSequence)))            
                f.write(" ".join(phoneStrings )+"\t "+" ".join(phoneStrings )+'\n')
                g.write(" ".join(map(str,item.sequence))+os.linesep)
            
def writeDict(phones,file):
    phoneStrings=list(map(lambda a:a.get()[0],phones))
    with open("data"+os.path.sep+file,'w') as f:
        for i,phone in enumerate(phoneStrings):
            f.write(str(i)+"\t"+" ".join(phone)+os.linesep)
            
writeDict(phones, "naive-phone.dict")
numSequences=3000
trainDataset=ToyAudioSequenceDataset(phones,len=numSequences,sequenceLengths=sequenceLengths,seed=5)
writeDataset(trainDataset,"naive-audio-train_{}.txt".format(numSequences))
 
numSequences=30000
trainDataset=ToyAudioSequenceDataset(phones,len=numSequences,sequenceLengths=sequenceLengths,seed=5)
writeDataset(trainDataset,"naive-audio-train_{}.txt".format(numSequences))
 
numSequences=300000
trainDataset=ToyAudioSequenceDataset(phones,len=numSequences,sequenceLengths=sequenceLengths,seed=5)
writeDataset(trainDataset,"naive-audio-train_{}.txt".format(numSequences))
 
 
 
numSequences=3000
valDataset=ToyAudioSequenceDataset(phones,len=numSequences,sequenceLengths=sequenceLengths,seed=6)
testDataset=ToyAudioSequenceDataset(phones,len=numSequences,sequenceLengths=sequenceLengths,seed=7)
             
writeDataset(valDataset,"naive-audio-test.txt")
writeDataset(testDataset,"naive-audio-val.txt")