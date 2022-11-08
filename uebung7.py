import recognizer.tools as tools
import numpy as np


if __name__ == "__main__":
    # Vektor der initialen Zustandswahrscheinlichkeiten
    logPi = tools.limLog([ 0.9, 0, 0.1 ])

    # Matrix der Zustandsübergangswahrscheinlichkeiten
    logA  = tools.limLog([
      [ 0.8,   0, 0.2 ], 
      [ 0.4, 0.4, 0.2 ], 
      [ 0.3, 0.2, 0.5 ] 
    ]) 

    # Beobachtungswahrscheinlichkeiten für "Regen", "Sonne", "Schnee" 
    # z=1 {  2: 0.1,  3: 0.1,  4: 0.2,  5: 0.5,  8: 0.1 },
    # z=2 { -1: 0.1,  1: 0.1,  8: 0.2, 10: 0.2, 15: 0.4 },
    # z=3 { -3: 0.2, -2: 0.0, -1: 0.8,  0: 0.0 }

    # gemessene Temperaturen (Beobachtungssequenz): [ 2, -1, 8, 8 ]
    # ergibt folgende Zustands-log-Likelihoods
    logLike = tools.limLog([
      [ 0.1,   0,   0 ],
      [   0, 0.1, 0.8 ],
      [ 0.1, 0.2,   0 ],
      [ 0.1, 0.2,   0 ]
    ])

    # erwartetes Ergebnis: ["1", "3", "2", "2"], 
    # pStar = log(9*16*4*8*1E-8) = -9.985131542
    print( tools.viterbi( logLike, logPi, logA ) )

    # verlängern der Beobachtungssequenz um eine weitere Beobachtung 
    # mit der gemessenen Temperatur 4
    # neue Beobachtungssequenz: [ 2, -1, 8, 8, 4 ]
    logLike = np.vstack( ( logLike, tools.limLog([ 0.2, 0, 0 ]) ) )

    # erwartetes Ergebnis: ["1", "3", "1", "1", "1"], 
    # pStar = log(9*16*3*8*16*1E-10) = -12.10539508
    print( tools.viterbi( logLike, logPi, logA ) )
