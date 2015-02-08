
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
	X = df.as_matrix(columns = df.columns[1:(np.size(df.columns)-1)])
	w0 = 2 * np.random.random_sample(np.size(X,axis=1)) + -1
	compute_y(w0,X[0,:])
	f = (df.ix[:,np.size(df.columns)-1]-2)/2
	f = f.as_matrix()
	error(f[1],w0,X[0,])

	#Test 4
	f = (df[,ncol(df)]-2)/2
	eta = 0.1
	max_iter = 100
	w = train_perceptron_sequential(X,f,max_iter,w0,eta)
	
def readIntoDf( dfA ):
	
	for i in range( 1 , np.size( dfA.columns ) ):
	
		column = dfA.ix[ : ,i ]
		nanArray =  np.where( np.isnan( column ) == True )
		
		for nanRow in nanArray:
		
			dfA.ix[ nanRow, i ] = column.median()
	
	return dfA
	
def activation(y):
	#Activation function
	threshold = 0.0
	
	if y > threshold:
		return 1
	else:
		return 0

def compute_y( w , x ):
	return np.dot( w ,  x )
	
def error(fy,w,x):
	return fy - activation( compute_y( w , x ) )
	

	

	

	
if __name__ == '__main__':
    main()