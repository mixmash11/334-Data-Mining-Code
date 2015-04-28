import pandas as pd
import numpy as np
import os
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier

class cKnn:
	
	def __init__( self, tr_data_in, tr_label_in, neighbors_in ):
		from sklearn.neighbors import KNeighborsClassifier
		# Set variables
		# self.tr_data = tr_data_in
		# self.tr_label = tr_label_in
		# neighbors = neighbors_in
		
		# Initialize classifier
		self.knnC = KNeighborsClassifier( n_neighbors=neighbors_in, weights='distance' )
		
		# Train data
		self.knnC.fit( tr_data_in, tr_label_in )
		# print self.knnC
	
	# Returns a score from given test data
	def score( self, te_data, te_label ):
		return self.knnC.score( te_data, te_label )

	def posScore( self, te_data, te_label ):
		posCount = 0
		total = 0
		for i in range( te_data.shape[0] ):
			if te_label[i] == 1:    
				total += 1
				if int(self.knnC.predict(te_data[i])) == 1:
					posCount += 1
		posScore = float( posCount / total )
		return posScore
	
def main():
	
		#Initialization
	os.chdir('C:\\Users\\Jesse\\Google Drive\\SPRING 2015\\CSCI 334\\334 Code\\Metabolite Project')
	
	dfP = pd.read_csv( 'positive_train2.csv', header=0, index_col=0 )
	dfN = pd.read_csv( 'negative_train2.csv', header=0, index_col=0 )

	dfP = dfP.transpose()
	dfP['Case'] = 1

	dfN = dfN.transpose()
	dfN['Case'] = 0

	bothDF = [ dfP, dfN ]
	dfC = pd.concat( bothDF )
	labels = dfC['Case']
	dfC = dfC.ix[:, :1095]

	#Making small classifiers
	samples = 20
	test = 10
	testEnd = samples + test

	dataPoints = 3

	dfPm = dfP.ix[0 : samples, 0 : dataPoints ]
	dfPmT = dfP.ix[ samples : testEnd, 0 : dataPoints ]
	dfNm = dfN.ix[0 : samples, 0 : dataPoints ]
	dfNmT = dfN.ix[ samples : testEnd, 0 : dataPoints ]
	dfNm['Case'] =0
	dfPm['Case'] =1
	dfNmT['Case'] =0
	dfPmT['Case'] =1

	dfCm = pd.concat( [ dfPm , dfNm ] )
	dfCmT = pd.concat( [ dfPmT , dfNmT ] )

	tr_data = np.array( dfCm.ix[ : , 0 : dataPoints ] )
	tr_label = np.array( dfCm.ix[ : , dataPoints :  ]  )

	te_data = np.array( dfCmT.ix[ : , 0 : dataPoints ] )
	te_label = np.array( dfCmT.ix[ : , dataPoints :  ]  )

	tr_label.resize( 40 )
	te_label.resize( 20 )

	knnC = KNeighborsClassifier( n_neighbors=5, weights='distance' )

	knnC.fit( tr_data, tr_label )
	print knnC.score( te_data, te_label )
	
def classTest():
	
	testC = customKnnC( tr_data, tr_label, 3 )
	testC.score( te_data, te_label )

	
if __name__ == '__main__':
    main()