import numpy as np
import recognizer.tools as tl
from sklearn.preprocessing import normalize

# default HMM
WORDS = {
    'name': ['sil', 'oh', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
    'size': [1, 3, 15, 12, 6, 9, 9, 9, 12, 15, 6, 9],
    'gram': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
}


class HMM:  

    words = {}

    def __init__(self, words=WORDS):
        """
        Constructor of HMM class. Inits with provided structure words
        :param input: word of the defined HMM.
        """
        self.words = words
        num_states = self.get_num_states()
        logPi = np.zeros(num_states)
        logA = np.zeros((num_states, num_states))
        size = self.words['size']
        startsi = np.cumsum(size) - 1
        startsj = np.roll(np.cumsum(size), 1)
        startsj[0] = 0

        iti = 0
        for i in range(num_states):
            # selbstuebergang
            logA[i][i] = 1
            # intra wort
            if i + 1 < num_states:
                logA[i][i+1] = 1
            # inter wort
            if i == startsi[iti]:
                itj = 0
                for j in range(num_states):
                    if j == startsj[itj]:
                        logA[i][j] = 1
                        if itj + 1 < len(startsj):
                            itj += 1
                logPi[i] = 1
                iti += 1
        self.logA = tl.limLog(normalize(logA, norm='l1', axis=1))
        self.logPi = tl.limLog(logPi/num_states)


    def get_num_states(self):
        """
        Returns the total number of states of the defined HMM.
        :return: number of states.
        """
        return sum(self.words['size'])

    def input_to_state(self, input):
        """
        Returns the state sequenze for a word.
        :param input: word of the defined HMM.
        :return: states of the word as a sequence.
        """
        if input not in self.words['name']:
            raise Exception('Undefined word/phone: {}'.format(input))

        # start index of each word
        start_idx = np.insert(np.cumsum(self.words['size']), 0, 0)

        # returns index for input's last state
        idx = self.words['name'].index(input) + 1

        start_state = start_idx[idx - 1]
        end_state = start_idx[idx]

        return [n for n in range(start_state, end_state) ]


    def getTranscription(self, statesequence):
        starts = np.roll(np.cumsum(self.words['size']), 1)
        starts[0] = -1

        wordsequence = []
        prev_state = -1
        for s in statesequence:
            if s in starts and s != prev_state:
                index = int(np.where(starts==s)[0])
                wordsequence.append(self.words['name'][index])
            prev_state = s
        return wordsequence
    
    def posteriors_to_transcription(self, posteriors):
        stateSequence, pStar = tl.viterbi(tl.limLog(posteriors), self.logPi, self.logA)
        return self.getTranscription(stateSequence)
