
import pandas as pd
import numpy as np
import os


def main():

	#Initialization
	os.chdir('C:\Users\Jesse\Google Drive\SPRING 2015\CSCI 334')
	df = pd.read_csv('breast-cancer-wisconsin.data', header=None, na_values=["?"] )
	df.columns = [ "Sample","Thickness","CellSize","CellShape","Adhesion","EpiCellSize", "Nuclei","Chromatin",'Nucleoli','Mitoses','Class' ]

	#Test for 1
	# print activation(0)
	# print activation(1)
	
	#Test 2
	df = readIntoDf(df)
	# print df
	
	#Test 3
	# X = df.as_matrix( columns = df.columns[ 1: ( np.size( df.columns ) -1 ) ] )
	# w0 = 2 * np.random.random_sample( np.size(X,axis=1) ) + -1
	# compute_y( w0 , X[ 0 , : ] )
	# f = ( df.ix[ : , np.size( df.columns ) -1 ] -2 ) / 2
	# f = f.as_matrix()
	# error( f[ 1 ], w0 , X[ 0 , ] )

	#Test 4
	# max_iter = 100
	# np.random.seed(100)
	# X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
	# w0 = 2 * np.random.random_sample(np.size(X,axis=1)) + -1
	# f = np.array([0,0,0,1])
	# eta = 0.1
	# print 'The input to this problem is: '
	# print X 
	# print 'The target for this problem is: '
	# print f
	# w = train_perceptron_sequential(X,f,max_iter,w0,eta)
	# print 'The result of w = train_perceptron_sequential(X,f,max_iter,w0,eta) is'
	# print w

	# Here is the output
	# The input to this problem is: 
	# [[1 0 0]
	 # [1 0 1]
	 # [1 1 0]
	 # [1 1 1]]
	# The target for this problem is: 
	# [0 0 0 1]
	# The result of w = train_perceptron_sequential(X,f,max_iter,w0,eta) is
	# [-0.31319012  0.25673877  0.14903518]

	# Here is test code for the breast cancer dataset
	# df = change_Class( df )
	# np.random.seed(100)
	# X = df.as_matrix(columns = df.columns[1:(np.size(df.columns)-1)])
	# w0 = 2 * np.random.random_sample(np.size(X,axis=1)) + -1
	# f = (df.ix[:,np.size(df.columns)-1]-2)/2
	# f = f.as_matrix()
	# eta = 0.1
	# max_iter = 100
	# w = train_perceptron_sequential(X,f,max_iter,w0,eta)
	# print "Weights of Data Set BC are:"
	# print w
	
	#Test 1, Cross Validation & Applying a Perception
	np.random.seed(100)
	df = pd.read_csv('breast-cancer-data-cleaned.csv')
	X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
	w0 = 2 * np.random.random_sample(np.size(X,axis=1)) + -1
	print('The results of predict(w0,X) ')
	print(predict(w0,X))
	
	
	#Test 2, Cross Validation & Applying a Perception
	df = pd.read_csv('breast-cancer-data-cleaned.csv')
	X = df.ix[1:100,2:(np.size(df,axis=1)-1)].as_matrix()
	w0 = 2 * np.random.random_sample(np.size(X,axis=1)) + -1
	eta = 0.1
	f = df.ix[1:100,np.size(df,axis=1)-1].as_matrix()/2-1
	max_iter=1
	acc = accuracy(X,f,w0,max_iter,eta)
	print 'The result of accuracy(X,f,w0,max.iter,eta) is'
	print acc
	
	
def readIntoDf( dfA ):
	
	for i in range( 1 , np.size( dfA.columns ) ):
	
		column = dfA.ix[ : ,i ]
		nanArray =  np.where( np.isnan( column ) == True )
		
		for nanRow in nanArray:
		
			dfA.ix[ nanRow, i ] = column.median()
	
	return dfA
	
def change_Class(dfA):
	for row in range( dfA.Class.count() ):
		if dfA.ix[ row , 'Class' ] == 2:
		
			dfA.ix[ row , 'Class' ] = 0
			
		else:
		
			dfA.ix[ row , 'Class' ] = 1
	
	return dfA
	
def activation(y):
	#Activation function
	threshold = 0.0
	
	if y > threshold:
		return 1
	else:
		return 0

def compute_y( w , x ):
	return np.dot( w ,  np.transpose( x ) )
	
def error(fy,w,x):
	return fy - activation( compute_y( w , x ) )
	
def train_perceptron_sequential(X,f,max_iter,w0,eta):
	# X = Matrix input
	# f = Vector containing targets
	# max_iter = Maximum iterations
	# w0 = Initial weights
	# eta = Value to adjust weights by
	
	w = w0
	numRows = X[:, 0 ].size
	update = 1
	
	for i in range( 0, max_iter ):
		if update == 0:
			return w
		else:
			update = 0
		
		for row in range( numRows ):
		
			#Saves w value to compare
			w_pre = w
			
			rowError = error( f[row], w, X[row] )
			
			w = update_rule( w, X[row], rowError, eta )
			
			if np.array_equal(w_pre, w) == False:
				update += 1
			
			
	
	
	return w

def update_rule( w, x, error, eta ):
	# w = vector of weights
	# x = vector of x values
	# error = computed error for the row
	# eta = adjustment
	
	scalar = error * eta
	
	if scalar == 0:
		return w
	
	else:
	
		x_error = scalar * x
		
		new_w = np.add( w , x_error )
		
		return new_w

def predict(w,X):

	if X.shape[0] == X.size:
		
		predOut = activation( compute_y( w, X ) )
		
		return predOut
		
	else:
		Xsize = X[ : , 0 ].size

		predOut = np.empty( Xsize )
		for row in range( predOut.size ):
			predOut[ row ] = activation( compute_y( w, X[row] ) )
	
		return predOut
	
def accuracy(X,f,w0,max_iter,eta):

	errors = 0.0
	numRows = X[:, 0].size
	
	
	for row in range( numRows ):
		
		Xtest = X[ row ]
		Xtrain = np.delete( X, row, 0 )
		ftrain = np.delete( f, row)
		
		weights = train_perceptron_sequential(Xtrain, ftrain, max_iter, w0, eta)
		
		fhat = predict(weights, Xtest)
		
		if fhat != f[row]:
			errors += 1
	
	return 1 - (errors / numRows )

if __name__ == '__main__':
    main()