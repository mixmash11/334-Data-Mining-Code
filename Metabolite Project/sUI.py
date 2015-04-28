# Simple User Interface 0.1
# A structured user interface for a Tyrosine Data Mining project

import pandas as pd
import numpy as np
import os
import sklearn


def set_input( min, max ) :
	
	user_input = 0
	while user_input > max or user_input < min:
		user_input = input("Enter a valid selection: ")
		print "\n"
		
	return user_input

def main():
	#Change this to your local data directory.
	os.chdir('C:\\Users\\Jesse\\Google Drive\\SPRING 2015\\CSCI 334\\334 Code\\Metabolite Project')

	print "Welcome to the Tyrosine Data Mining Tool\nVersion 0.1\n"

	# Section 1: Select Data
	print "Section 1: Select Data ---------- \n"
	print "Choose from the following pieces of data:"
	print "1) negative_train2.csv"
	print "2) negative_train5.csv"
	print "3) negative_train10.csv"

	input1 = set_input( 1 , 3 )
	if input1 == 1:
		dfP = pd.read_csv( 'positive_train2.csv', header=0, index_col=0 )
		dfN = pd.read_csv( 'negative_train2.csv', header=0, index_col=0 )
	elif input1 == 2:
		dfP = pd.read_csv( 'positive_train5.csv', header=0, index_col=0 )
		dfN = pd.read_csv( 'negative_train5.csv', header=0, index_col=0 )
	elif input1 == 3:
		dfP = pd.read_csv( 'positive_train10.csv', header=0, index_col=0 )
		dfN = pd.read_csv( 'negative_train10.csv', header=0, index_col=0 )

		# Basic data processing operations
	dfP = dfP.transpose()
	dfP['Case'] = 1
	dfN = dfN.transpose()
	dfN['Case'] = 0
	bothDF = [ dfP, dfN ]
	dfC = pd.concat( bothDF )
	labels = dfC['Case']
	dfC = dfC.ix[:, :1095]


	# Section 2: Preprocess Data
	print "\nSection 2: Preprocess Data ---------- \n"
	while True:
		print "Choose from the following Methods:"
		print "1) Cross validation data separation"
		print "2) Move to Next Step"

		input2 = set_input( 1 , 2 )
		if input2 == 1:
			from sklearn.cross_validation import train_test_split
			
			df_train, df_test, label_train, label_test = train_test_split( dfC, labels )
			tr_data = np.array( df_train )
			tr_label = np.array( label_train )
			te_data = np.array( df_test )
			te_label = np.array( label_test )
			
		elif input2 == 2:
			break
			
	# Section 3: Evaluation Methods
	print "\nSection 3: Evaluation Methods ---------- \n"

	#Array for later tests
	section3tests = []
		
	while True:
		itemNum = (len(section3tests) + 1)
		print "Item %d-------" % itemNum
		print "Choose from the following Methods:"
		print "1) K nearest-neighbor"
		print "2) Support Vector Machine"
		print "3) Move to Next Step"
		
		

		input3 = set_input( 1 , 3 )
		if input3 == 1:
			import knnClassifier as knnP
			
			section3tests.append(1)
			
		elif input3 == 2:
			import svmClassifier as svmP
			
			section3tests.append(2)
			
		elif input3 == 3:
			break
			
	# Section 4: Method Conditionals
	print "\nSection 4: Method Conditionals ---------- \n"



		# Making vectors in a matrix for the tests, ordered by row
	algVars = 2
	specVars = 2
	sumVars = 1 + algVars + specVars #Sum of features per vector, will be dimension of vector
	executionVectors = np.zeros( ( len( section3tests ) , sumVars ), dtype=int )

	for i in range( len( section3tests ) ):
		itemNum = i + 1
		print "Item %d-----" % itemNum
		
		#Algorithm Specific Conditions
		if section3tests[i] == 1: #K nearest-neighbor
			print "K nearest-neighbor Settings:"
			
			executionVectors[i, 0] = 1 #Slot 1 - Algorithm
			print "Neighbors (3-10):"
			executionVectors[i, 1] = set_input(3, 10) #Slot 2 - Neighbors
			
			
		elif section3tests[i] == 2: #SVM
			print "Support vector machine Settings:"
			
			executionVectors[i, 0] = 2 #Slot 1 - Algorithm
			print "Kernel:"
			print "     1) Linear"
			print "     2) Radial Basis"
			executionVectors[i, 1] = set_input(1, 2) #Slot 2 - Neighbors
		
		while True:
			print "Choose from the following Customizations:"
			print "1) Move to Next Step"
			print "2) Find best data points"
			
			

			input4 = set_input( 1 , 2 )
			if input4 == 1:
				break
				
			elif input4 == 2:
				executionVectors[i, 4] = 1

	# Section 5: Execution
	print "\nSection 5: Execution ---------- \n"

	dataMatrix = np.zeros( (executionVectors.shape[0], algVars + 2)  )

	for i in range( executionVectors.shape[0] ):
		
		if executionVectors[i, 4] != 1:
			
			if executionVectors[i, 0] == 1: # K nearest-neighbor
				neighbors = executionVectors[i, 1]
				
				classifierK = knnP.cKnn( tr_data, tr_label, neighbors )
				
				dataMatrix[i, 0] = classifierK.score( te_data, te_label )
				dataMatrix[i, 1] = 1
				dataMatrix[i, 2] = neighbors
				dataMatrix[i, 3] = 0
			
			if executionVectors[i, 0] == 2: # SVM
				
				if executionVectors[i, 1] == 1:
					kernel1 = 'linear'
				elif executionVectors[i, 1] == 2:
					kernel1 = 'rbf'
				
				classifierS = svmP.cSVM( tr_data, tr_label, kernel1 )
				
				dataMatrix[i, 0] = classifierS.score( te_data, te_label )
				dataMatrix[i, 1] = 1
				dataMatrix[i, 2] = kernel1
				dataMatrix[i, 3] = 0
				
		else: # Special condition, Best data points
			print "word"
	print "Processing Complete"

	# Section 6: Results
	print "\nSection 6: Results ---------- \n"

	headers = ["Score",  "Algorithm", "Modifier", "None" ]

	resultDf = pd.DataFrame( dataMatrix, columns=headers )

	print "End of Program"

if __name__ == '__main__':
    main()
	
	
# Section Template
	
# Section TK: FUNCTION
	# print "\nSection TK: FUNCTION \n"
	# while True:
		# print "Choose from the following Methods:"
		# print "1) TASK"
		# print "2) Move to Next Step"

		# inputTK = set_input( 1 , 2 )
		# if inputTK == 1:
			
			
			
		# elif inputTK == 2:
			# break
