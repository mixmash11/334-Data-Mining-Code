import pandas as pd
import numpy as np
import os
import sklearn as sk
from sklearn import svm

class cSVM:
	def __init__( self, tr_data_in, tr_label_in, kernel_in ):
		from sklearn import svm
		#Initialize SVM classifier
		self.svmC = svm.SVC(kernel=kernel_in)
		
		#Fit to training data
		self.svmC.fit( tr_data_in, tr_label_in )
		
		# print self.svmC
		
	# Returns a score from given test data	
	def score( self, te_data, te_label ):
		return self.svmC.score( te_data, te_label )
		
	def posScore( self, te_data, te_label ):
		posCount = 0
		total = 0
		for i in range( te_data.shape[0] ):
			if te_label[i] == 1:    
				total += 1
				if int(self.svmC.predict(te_data[i])) == 1:
					posCount += 1
		posScore = float( posCount / total )

		return posScore

def classTest():
	
	testC = customSVMc( tr_data, tr_label )
	testC.score( te_data, te_label )
		