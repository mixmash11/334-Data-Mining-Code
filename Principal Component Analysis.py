
import pandas as pd
import numpy as np
import os

# Given Description:
# In this lesson you will implement the fundamentals of principal component analysis (PCA) for a dataset that has two dimensions. This is not how a general purpose PCA algorithm would be created. It is for instructional purposes only. Specifically, we will be finding the axis of greatest variance, and we'll call this PC1 for the first principal component. After this is found, we will use it to project our original dataset onto this new dimension.


def main():

	#Test 1
	X = np.array([[1,1],[2,1],[3,1],[4,2]])
	print 'The result of covariance_matrix(X) is'
	print covariance_matrix(X)

	np.random.seed(100)
	X = np.random.random_sample([10,2])
	print 'The result of covariance_matrix(X) is'
	print covariance_matrix(X)
	
	#Test 2
	C = np.array([[8.39,7.6],[7.6,7.71]])
	print 'The result of eign_values(C) is'
	print eigen_values(C)
	
	#Test 3
	print 'The result of cumulative_percent_variance_explained(np.array([1,0.5,.2])) is'
	print cumulative_percent_variance_explained(np.array([1,0.5,.2]))
	
	#Test 4
	C = np.array([[ 8.39,  7.6 ],[ 7.6 ,  7.71]])
	eigen_value = 15.66
	PC1 = eigen_vector(eigen_value,C)
	print 'The value of PC1 is'
	print PC1
	
	#Test 5
	np.random.seed(100)
	X = np.random.random_sample([10,2])
	C = covariance_matrix(X)
	values = eigen_values(C)
	PC1 = eigen_vector(values[0],C)
	scores = np.dot(X,PC1)
	print 'The values in the new dimension are'
	print scores
	
def covariance_matrix(X):
	
	# To begin, we will want to implement a function to compute the covariance matrix for a N x 2 dataset, X. 
	#The covariance matrix, C, for this size of dataset will be 2 x 2.
	
	C = np.zeros( [ 2,2 ] , dtype=float )
	
	mean = np.mean( X )
	
	N = X[ : , 0].size
	
	for row in range( 2 ):
		
		for col in range( 2 ):
		
			sum = float( 0 )
			
			for k in range( N ):
				
				sum += ( X[ k, row ] - np.mean( X , 0 )[row] ) * ( X[k, col]  - np.mean( X , 0 )[col] ) / (N - 1)
		
			C[ row , col ] = sum
	
	return C
	
def eigen_values(C):

	#Now that you have the covariance matrix, the next step is to determine the eigen values for the covariance matrix, C
	#Specifically, your function should return the eigen values in sorted order (descending).
	
	a = C[0,0]
	b = C[0,1]
	d = C[1,1]

	eV = np.empty( 2 )

	disc = np.sqrt( ( a - d ) * ( a - d ) + 4*b*b ) / 2
	lambda1 = ( a + d ) / 2 + disc
	lambda2 = ( a + d ) / 2 - disc

	if lambda1 > lambda2:
		eV[0] = lambda1
		eV[1] = lambda2
	else:
		eV[0] = lambda2
		eV[1] = lambda1

	return eV
	
def cumulative_percent_variance_explained(lambdas):
	
	#Now we will write a function that returns the cumulative percent variance explained by the principal components using the eigenvalues. 
	#This can be found by finding the cumulative sum of the eigenvalues which should be in descending order and then dividing by the sum of all lambdas.
	
	eVsum = np.sum( lambdas )
	cpvE = np.empty( lambdas.size )
	for i in range( lambdas.size ):
		
		cpvE[ i ] = (lambdas[ i ] / eVsum)
		
		if i > 0:
			cpvE[ i ] += cpvE[ i - 1 ]
	
	return cpvE
	
def eigen_vector(eigen_value,C):

	eVec = np.empty( 2 )

	lambdaIden = np.dot(np.identity(2), eigen_value)
	
	CminusLambda = np.subtract( C, lambdaIden )
	
	magE = np.sqrt( CminusLambda[0,0] * CminusLambda[0,0] + CminusLambda[0,1] * CminusLambda[0,1])
	
	if CminusLambda[ 0, 0 ] > CminusLambda[ 0, 1 ]:
		eVec[ 0 ] = np.absolute(CminusLambda[ 0, 0 ] / magE)
		eVec[ 1 ] = np.absolute(CminusLambda[ 0, 1 ] / magE)
	else:
		eVec[ 0 ] = np.absolute(CminusLambda[ 0, 1 ] / magE)
		eVec[ 1 ] = np.absolute(CminusLambda[ 0, 0 ] / magE)

	return eVec
	
if __name__ == '__main__':
    main()