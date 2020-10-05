"""
    Code to compose music 
"""

# Libraries #
import pickle
import glob
import os
import numpy as np
import music21 as m21
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM, Activation #, LSTM #CuDNNLSTM for Nvidia graphics card
from keras.optimizers import adam
import time

def main():
    # Get the genre from user input
    print("Hello, let's get started by picking a genre for me to compose")
    genre = input("I can currently compose Jazz, Pop or IrishFolk... Please enter a genre as they appear in this list... ")
    genre = CheckGenre(genre)

    # Get external data 
    fileHandler = FileHandler(genre)
    notesData = fileHandler.GetNotesData()
    networkWeights = fileHandler.GetWeights()

    print(networkWeights)

    # Prepare data to give to the network
    dataPrep = DataPrep(notesData)
    vocabLen = dataPrep.GetVocabLen()

    availablePitches = dataPrep.GetGroupedPitches()

    notesDict = dataPrep.DictionarisePitches()
    netInput, normInput = dataPrep.CreateSequences()

    # Create the model
    model = ModelDef(networkWeights, normInput, vocabLen)
    nn = model.BuildModel()

    # Compose the song
    composer = Composer(nn, netInput, availablePitches, vocabLen)
    composition = composer.ComposeSong()

    midi = composer.CreateMidi()
    title = input("Please provide a name for this new composition... ")

    # Write song to midi file to directory 
    fileHandler.WriteMidiToDir(title, midi)


def CheckGenre(genre):
    while genre != 'Jazz' and genre != 'Pop' and genre != 'IrishFolk':
        print("Sorry I don't currently know that genre!")
        genre = input("Please choose between Jazz, Pop or IrishFolk. Please enter the genre as they appear in this list... ")

    return genre


############ Classes ############
class FileHandler:
    #Attributes


    #Constructor
    def __init__(self, genre):
        self._genre = genre

    #Properties


    #Methods
    def GetNotesData(self):
        print("Collecting data from the " + self._genre + " notes data page")
        filePath = "NotesData/" + self._genre + "Notes/NotesPage"

        pickleFile = open(filePath, 'rb')
        notesData = pickle.load(pickleFile) 
        pickleFile.close()

        return notesData

    def GetWeights(self):
        print("Collecting the weights for the neural network")
        # Loop through directory collecting all files (they will all be weights)
        filePaths = "Weights/" + self._genre + "Weights/*"

        # Get the latest file to be inserted, this will be the best
        weightsList = sorted(glob.iglob(filePaths), key=os.path.getmtime, reverse=True)
        bestWeights = weightsList[0]

        return bestWeights

    def WriteMidiToDir(self, fileName, midiData):
        fileName = fileName + ".mid"
        midiData.write('midi', fp=fileName)



class DataPrep:
    #Attributes
    vocabLen = None
    sequenceLen = 40
    groupedPitches = []
    pitchesDictionary = {}
    inputSequences = []
    outputForSequences = []

    #Constructor
    def __init__(self, notes):
        self._notes = notes

    #Properties
    def GetNotes(self):
        return self._notes

    #Methods
    def GetVocabLen(self):
        self.vocabLen = len(set(self._notes))

        return self.vocabLen

    def GetGroupedPitches(self):
        self.groupedPitches = sorted(set(self._notes))

        return self.groupedPitches

    def DictionarisePitches(self):
        for i, pitch in enumerate(self.groupedPitches):
            self.pitchesDictionary.update({pitch : i})

        return self.pitchesDictionary

    def CreateSequences(self):
        for i in range (0, len(self._notes) - self.sequenceLen, 1):
            # Create a list of input sequences of notes and chords
            inputSeq = self._notes[i:i + self.sequenceLen]
            # Get the output (next note / chord) that appears after the sequences 
            output = self._notes[i + self.sequenceLen]
            # Get the integer representation of the notes / chords for the input sequences - based on position in the pitch dictionary
            self.inputSequences.append([self.pitchesDictionary[i] for i in inputSeq])
            # Get the integer representation of the outputs for the sequences - based on position in the pitch dictionary
            self.outputForSequences.append(self.pitchesDictionary[output])
            
        # Get the number of patters 
        numberOfPatterns = len(self.inputSequences) #2520 sequences

        # Reshape 
        normalizedInput = np.reshape(self.inputSequences, (numberOfPatterns, self.sequenceLen, 1))

        normalizedInput = normalizedInput / float(self.vocabLen)

        return self.inputSequences, normalizedInput


        
class ModelDef:
    #Attributes
    model = Sequential()
    opt = adam(lr=0.0001, decay=1e-6)

    #Constructor
    def __init__(self, weights, normInput, vocabLen):
        self._weights = weights
        self._normInput = normInput
        self._vocabLen = vocabLen


    #Properties

    
    #Methods
    def BuildModel(self):
        self.model.add(LSTM(
            512,
            input_shape=(self._normInput.shape[1], self._normInput.shape[2]),
            return_sequences=True
        ))

        # Recurrent layers
        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(LSTM(512))

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

        self.model.load_weights(self._weights)

        return self.model



class Composer:
    #Attributes
    compLen = 500
    predictedNotes = []
    offset = 0
    newDict = {}
    notesForMidi = []
    instrument = m21.instrument.Piano()

    #Constructor
    def __init__(self, model, netInput, pitches, vocabLen):
        self._nn = model
        self._netInput = netInput
        self._pitches = pitches
        self._vocabLen = vocabLen

    #Properties


    #Methods
    def NewDictionary(self):
        # New format for dictionary
        for i, pitch in enumerate(self._pitches):
            self.newDict.update({i : pitch})

        return self.newDict

    def ComposeSong(self):
        # Get the number : note dictionary
        newDict = self.NewDictionary()

        # Create a random seed to start the composition 
        startNotes = np.random.randint(0, len(self._netInput) - 1)
        
        # Get the sequence from the random int 
        inputPattern = self._netInput[startNotes]

        for i in range(self.compLen):
            predSeq = np.reshape(inputPattern, (1, len(inputPattern), 1))
            predSeq = predSeq / float(self._vocabLen)

            newPrediction = self._nn.predict(predSeq, verbose=1)

            index = np.argmax(newPrediction)
            res = newDict[index]
            self.predictedNotes.append(res)

            inputPattern.append(index)
            inputPattern = inputPattern[1:len(inputPattern)]

        return self.predictedNotes

    def CreateMidi(self):
        # Loop through the list of notes
        for i in self.predictedNotes:
            # Check if the notes are a collection to make a chord, if it contains a full stop it's a chord
            if ('.' in i):
                #Store the notes as seperate values in list
                notesCol = i.split('.')
                for j in notesCol:
                    note = note.Note(int(j))
                    note.storedInstrument = self.instrument
                chord = m21.chord.Chord(notesCol)
                chord.offset = self.offset
                self.notesForMidi.append(chord)
            else:
                note = m21.note.Note(i)
                note.offset = self.offset
                note.storedInstrument = self.instrument
                self.notesForMidi.append(note)
            
            self.offset += 0.5

        

        midi = m21.stream.Stream(self.notesForMidi)

        return midi


if __name__=='__main__':
    main()