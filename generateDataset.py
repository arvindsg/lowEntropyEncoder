from toydataset import ToyAudioSequenceDataset, NoiseFreeSymbol
numphones=30
phones=[NoiseFreeSymbol(channels=1) for _ in range(numphones)]
numSequences=30000
sequenceLengths=10
trainDataset=ToyAudioSequenceDataset(phones,len=numphones,sequenceLengths=sequenceLengths,seed=5)
valDataset=ToyAudioSequenceDataset(phones,len=numphones,sequenceLengths=sequenceLengths,seed=6)
testDataset=ToyAudioSequenceDataset(phones,len=numphones,sequenceLengths=sequenceLengths,seed=7)
def writeDataset(dataset,file):
    with open(file,'w') as f:
        for item in dataset:
            print(item)
        
writeDataset(trainDataset,"naive-audio-train.txt")