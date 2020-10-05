""" 
    Train the network
"""

# Libraries
import glob
import pickle
import numpy as np
import tensorflow as tf
import music21 as m21
from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense, Activation, BatchNormalization#, CuDNNLSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
import time


def main():

    print("Hello. Let's get started by picking a type of music to train with")
    print("Please pick a genre of music you would like me to learn.")

    userInput = input("Please type Jazz, Pop or IrishFolk as they appepar in this list... ")
    genre = UserInput(userInput)

    ############# Get training data from the chosen directory ############
    fileHandler = FileHandler(genre)
    midiData = fileHandler.GetMidiFromDirectory()
    #print(midiData)

    ############# Pre-processing for dataset ############
    dataPreProcess = PreProcess(midiData)

    #Convert songs to stream object
    streamObjList = dataPreProcess.ConvertDataToStream()

    #Transpose songs
    transposedSongList = dataPreProcess.TransposeMusic(streamObjList)

    #Extract notes from score object
    songNotesData = dataPreProcess.ExtractNotesFromScore()
    #print(f"song Data {(songNotesData)}")

    #Pickle notes
    fileHandler.PickleNotes(songNotesData)

    #Get length of vocab
    dataPrep = DataPrep(songNotesData)
    vocabLen = dataPrep.getVocabLen()
    #Group together pitches
    groupedPitches = dataPrep.GroupPitches(songNotesData)
    dictNotes = dataPrep.DictionariseNotes()

    #Create training sequences
    networkInput, networkDesiredOutput = dataPrep.CreateTrainingSequences()

    ############ Create the shape of the model ############
    modelShape = ModelDefinition(networkInput, vocabLen)
    model = modelShape.DefineModel()

    ########### Train the model ############
    TrainNetwork(model, networkInput, networkDesiredOutput, genre)


def UserInput(genre):
    while genre != 'Jazz' and genre != 'Pop' and genre != 'IrishFolk':
        print("Sorry I haven't got the resources to learn that style yet!")
        genre = input("Please choose between Jazz, Pop or IrishFolk as they appear in this list... ")

    return genre


def TrainNetwork(model, networkInput, networkDesiredOutput, genre):
    filePath = './Weights/' + genre + 'Weights/Epoch{epoch:02d}-Loss{loss:.5f}'
    print(filePath)
    checkPoint = ModelCheckpoint(filePath, monitor='loss', verbose=0, save_best_only=True, mode='min')

    callbackList = [checkPoint]

    model.fit(networkInput, networkDesiredOutput, epochs=40000, batch_size=64, callbacks=callbackList)


############ Classes ############
# This class will be used to obtain the data and write to external files #

class FileHandler:
    #Attributes
    midiSongList = []

    #Constructor
    def __init__(self, genre):
        self._genre = genre

    #Properties

    #Methods
    def GetMidiFromDirectory(self):
        print("Getting data from the " + self._genre + " midi dataset")

        for i in glob.iglob("MidiData/" + self._genre + "Midi/*.mid"):
            self.midiSongList.append(i)

        return self.midiSongList


    def PickleNotes(self, songNotesData):
        #Pickle data to external file
        notesFile = open('./NotesData/' + self._genre + 'Notes/NotesPage', 'wb')
        pickle.dump(songNotesData, notesFile)
        notesFile.close()


###### Data Pre-Processing Class ######
# This class will be used to pre-process the data #

class PreProcess:
    #Attributes
    songListCon = []
    songListTransposed = []
    songDataNotes = []

    #Constructor
    def __init__(self, midiData):
        self._midi = midiData

    #Properties

    #Methods
    def ConvertDataToStream(self):
        print("Converting songs... this could take a few minutes depending on the size of the dataset")
        
        for i in self._midi:
            #convert songs to stream
            convData = m21.converter.parse(i)
            self.songListCon.append(convData)

        print("I have finished converting the songs")

        return self.songListCon


    def TransposeMusic(self, streamObjList):
        print("Transposing songs to key of C major or A minor depending on it's current mode.This could take a few minutes depending on the size of the dataset")

        # Loop through dataset and transpose songs from C major or A minor
        for i in streamObjList:
            # Find current key of an instance
            currentKey = i.analyze('key')

            if currentKey.mode == 'major':
                #Find distance between current key and C major
                distanceToKey = m21.interval.Interval(currentKey.tonic, m21.pitch.Pitch('C'))
            elif currentKey.mode == 'minor':
                #Find distance between current key and A minor
                distanceToKey = m21.interval.Interval(currentKey.tonic, m21.pitch.Pitch('A'))
            else:
                print("Couldn't obtain the mode of the song. Removing sample to reduce potential confusion")
                self.songListCon.remove(i)

            #Transposed the song by the distance calculated
            transposedSong = i.transpose(distanceToKey)

            self.songListTransposed.append(transposedSong)

        print("All songs have now been transposed")

        return self.songListTransposed


    def ExtractNotesFromScore(self):
        
        for song in self.songListTransposed:
            #Convert each song to a flat file structure and filter out anything that isn't a note or a rest
            songFlatStruct = song.flat.notesAndRests

            # Loop through track and check for all notes and chords
            for note in songFlatStruct:
                if type(note) is m21.note.Note:
                    self.songDataNotes.append(str(note.pitch))
                elif type(note) is m21.chord.Chord:
                    #iterate through notes and join to make chord
                    self.songDataNotes.append('.'.join(str(notes) for notes in note.normalOrder))

            return self.songDataNotes

                
###### Data Preperation Class ######
# This class will be used to get the data ready to be fed into the neural network #

class DataPrep:
    #Attributes
    vocabLen = None
    sequenceLen = 64
    groupedPitches = []
    indexedPitches = {}
    inputSequences = []
    targetOutput = []
    networkInput = []
    networkDesiredOutput = []

    #Constructor
    def __init__(self, notesData):
        self._notesData = notesData

    #Properites
    def getVocabLen(self):
        self.vocabLen = len(set(self._notesData))

        return self.vocabLen

    #Methods
    def GroupPitches(self, songNotesData):
        #Group pitches together
        self.groupedPitches = sorted(set(songNotesData))

    def DictionariseNotes(self):
        #Create a dictionary to store notes/ chords with a respective index value
        for i, pitch in enumerate(self.groupedPitches):
            #Append index and pitch to dictionary
            self.indexedPitches.update({pitch : i})

        return self.indexedPitches

    def CreateTrainingSequences(self):
        #Create training sequences and their desired output #2560
        for i in range (len(self._notesData) - self.sequenceLen):
            #Create input sequences of 64 notes as i increments it will shift >>
            inputSequences = self._notesData[i:i + self.sequenceLen]
            #Create the desired output for the sequence i.e. the next note of the sequence
            desiredOutput = self._notesData[self.sequenceLen + i]

            #Create the dictionary representation of input sequences
            self.networkInput.append([self.indexedPitches[char] for char in inputSequences])

            #Store the dictionary representation for the desire outputs
            self.networkDesiredOutput.append(self.indexedPitches[desiredOutput])

        numberofPatterns = len(self.networkInput)

        self.networkInput = np.reshape(self.networkInput, (numberofPatterns, self.sequenceLen, 1))

        self.networkInput = self.networkInput / float(self.vocabLen)

        #one-hot-encode
        self.networkDesiredOutput = np_utils.to_categorical(self.networkDesiredOutput)

        return self.networkInput ,self.networkDesiredOutput

       

###### Model Definition Class ######
# This class will be used to define the shape of the model by adding LSTM (CuDNNLSTM layers for GPU, activation functions and optimization layers) #
class ModelDefinition:
    #Attributes
    model = Sequential()
    opt = adam(lr=0.0001, decay=1e-6)

    #Constructor
    def __init__(self, networkInput, vocabLen):
        self._networkInput = networkInput
        self._vocabLen = vocabLen

    #Properties


    #Methods

    def DefineModel(self):
        # Input layer
        self.model.add(CuDNNLSTM(
            512,
            input_shape=(self._networkInput.shape[1], self._networkInput.shape[2]),
            return_sequences=True
        ))

        # Recurrent layers
        self.model.add(CuDNNLSTM(512, return_sequences=True,))
        self.model.add(CuDNNLSTM(512))

        # Transorm data to be between 0 and 1
        self.model.add(BatchNormalization())

        # Ignore units to help prevent overfitting
        self.model.add(Dropout(0.3))

        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        self.model.add(Dropout(0.3))
        self.model.add(Dense(self._vocabLen))

        self.model.add(Activation('softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer=self.opt)

        return self.model



if __name__ == '__main__':
    main()