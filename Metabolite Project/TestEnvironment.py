import pandas as pd
import numpy as np
import os
import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


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

		#Cross-validation set up
	df_train, df_test, label_train, label_test = train_test_split( dfC, labels )

	a_train = np.array( df_train )
	l_train = np.array( label_train )
	a_test = np.array( df_test )
	l_test = np.array( label_test )
	
		#Adjust size for testing
	a_train = a_train[:, :3]
	a_test = a_test[:, :3]

	
		#Generic SVM vs. K-nearest Neighbor Test
	svmC = svm.SVC()
	knnC = KNeighborsClassifier(n_neighbors=5)
	
	svmC.fit( a_train, l_train )
	knnC.fit( a_train, l_train )
	
	svmCount = 0
	knnCount = 0
	
	for i in range( 20 ):
		test = a_test[ i ]
		key = l_test[ i ]
		
		sOut = svmC.predict( test )
		kOut = svmC.predict( test )
		print "Key: "
		print key
		print "SVM:"
		print sOut
		print "KNN:"
		print kOut
		
		if int(sOut) == key:
			svmCount += 1
		if int(kOut) == key:
			knnCount += 1
	
	svmAcc = svmCount / a_test[:, 0].size
	knnAcc = knnCount / a_test[:, 0].size
	
	print "SVM Accuracy:"
	print svmAcc
	print "KNN Accuracy"
	print knnAcc
	
	
if __name__ == '__main__':
    main()