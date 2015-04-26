import pandas as pd
import numpy as np
import os
import sklearn as sk
from sklearn import svm

class customSVMc:
	def __init__( self, tr_data_in, tr_label_in ):
		from sklearn import svm
		#Initialize SVM classifier
		self.svmC = svm.SVC(kernel='linear')
		
		#Fit to training data
		self.svmC.fit( tr_data_in, tr_label_in )
		
		# print self.svmC
		
	# Returns a score from given test data	
	def score( self, te_data, te_label ):
		return self.svmC.score( te_data, te_label )
		
def classTest():
	
	testC = customSVMc( tr_data, tr_label )
	testC.score( te_data, te_label )
		