import recognizer.hmm as HMM
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # default HMM
    hmm = HMM.HMM()
    
    # statesequence = [ 0, 1, 1, 2, 2, 3]                                # oh
    # statesequence = [ 1, 2, 3,3,31,32,33,34,35,36,36,1,2,3,0]          # oh two oh
    statesequence = [ 31,32,33,34,35,36,0,1,2,3,1,2,3,31,32,33,34,35,36] # two oh oh two 
    #print( hmm.logA )


    words = hmm.getTranscription(statesequence)
    print(words) # ['two', 'oh', 'oh', 'two']

    plt.imshow(np.exp(hmm.logA))
    plt.xlabel('in Zustand j')
    plt.ylabel('von Zustand i')
    plt.colorbar()
    plt.clim(0.0, 0.2)
    plt.show()
