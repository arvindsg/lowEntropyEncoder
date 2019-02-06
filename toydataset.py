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

class Lang(object):
    
    def __init__(self,buildLangs=True,word_lang=None,char_lang=None,output_lang=None,dataset=None,maxVocabSamples=10000):
        if not buildLangs:
            assert word_lang is not None and char_lang is not None and output_lang is not None
        else:
            assert dataset is not None
            #build langs
        self.maxVocabSamples=maxVocabSamples    

        word_lang,char_lang=self.buildDialogLang(dataset)
        self.word_lang=word_lang
        self.char_lang=char_lang
        self.output_lang=word_lang
        self.indexStore=None
    def updateLang(self,store,word_lang,char_lang):
        pass
    def buildKBLang(self,dataset,lang=None):
        if lang is None:
            lang=Lang()
        samples=min(self.maxVocabSamples,dataset.__len__())
        for i in range(samples):
            sample=dataset.__getitem__(i)
            store=sample["store"]
            for row in store:
                lang.index_words(" ".join(row.values()))
        return lang
    def buildDialogLang(self,dataset):
        word_lang=Lang()
        char_lang=Lang()
        
        samples=min(self.maxVocabSamples,dataset.__len__())
        for i in range(samples):
            sample=dataset.__getitem__(i)
            dialog=sample["dialog"]
            for turn in dialog:
                
                text=" ".join(turn)                    
                word_lang.index_words(normalize_string(text))
                words=text.split()
                for word in words:
                    char_lang.index_words(" ".join(get_shortened_word(word)))
        word_lang=self.buildKBLang(dataset, word_lang)      
        return word_lang,char_lang        
        
                    
    def __call__(self,sample):
        #use langs to convert variables to tensors
        context=sample["context"]
        joinedContext=""
        charIndexContext=[]
        for line in context:
            joinedContext+=line
            joinedContext+=" "+EOS_label+" "
            words=line.split(" ")
            for word in words:
                charIndexContext.append(indexes_from_sentence(self.char_lang, " ".join(get_shortened_word(word)))[:-1])
            charIndexContext.append([EOS_token])
        sample["indexContext"]=indexes_from_sentence(self.word_lang,joinedContext.strip())[:-1]
        sample["indexTarget"]=indexes_from_sentence(self.word_lang,sample["target"])
        sample["charIndexContext"]=charIndexContext
        if sameStoreForAllSamples and self.indexStore is not None:
            sample["indexStore"]=self.indexStore
        else:
            self.indexStore=[[indexes_from_sentence(self.word_lang,value)[:-1] for value in row.values()] for row in sample["store"]]
            sample["indexStore"]=self.indexStore
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
