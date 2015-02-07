
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
	

	

def activation(y):
	#Activation function
	threshold = 0.0
	
	if y > threshold:
		return 1
	else:
		return 0

def readIntoDf( dfA ):
	
	for i in range( 1 , np.size( dfA.columns ) ):
	
		column = dfA.ix[ : ,i ]
		nanArray =  np.where( np.isnan( column ) == True )
		
		for nanRow in nanArray:
		
			dfA.ix[ nanRow, i ] = column.median()
	
	return dfA
	
if __name__ == '__main__':
    main()