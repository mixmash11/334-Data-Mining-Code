
import pandas as pd
import numpy as np
import os

def main():
	os.chdir('C:\Users\Jesse\Google Drive\SPRING 2015\CSCI 334')
	df = pd.read_csv('train.csv')

	# Test for problem 1
	# df.Fare = np.round(df.Fare/10)
	# df.Age = np.round(df.Age/10)
	# print compute_priors(df.Survived,np.array([0,1]))

	#Test for problem 2
	# print compute_likelihood(df.Survived,0,df.Sex,'female')
	# print compute_likelihood(df.Survived,0,df.Sex,'male')

	#Test for problem 3
	# print compute_likelihood_data(df.Survived,1,df.ix[:,[5,6,7]],df.ix[0,[5,6,7]])

	#Test for problem 4
	# print compute_posterior(df.Survived,np.array([0,1]),df.ix[:,[5,6,7]],df.ix[0,[5,6,7]])


def compute_priors(labels,classes):
	# The parameter labels contains the values that we are hoping to predict for all of the training samples (e.g., df$Survived or df.Survived). The classes parameter has the unique values of the labels (e.g., c(0,1) or np.array([0,1])).
    
    
    classSize = classes.size
    labelsSize = float( labels.size )
    
    priors = np.empty(classSize, dtype=float)
    
    ##print labelsSize
    
    for entrys in range(classSize):
        
		priors[ entrys ] = np.asarray( np.where( labels == classes[entrys] ) ).size / labelsSize
            
    return priors

def compute_likelihood(labels,class_value,values,value):
	# The parameter labels is the same as the previous function. The parameter class is now a specific label value (e.g., 0). The values parameter is a column of the dataframe (e.g., df$Sex or df.Sex), and value is a specific value of that variable (e.g., 'male').
	
	numerator = float( np.asarray(np.where( np.logical_and(labels == class_value, values == value) )).size )
	denominator = float( np.asarray(np.where( labels == class_value ) ).size )
	
	p = numerator / denominator
		
	return p
	
def compute_likelihood_data(labels,class_value,data,evidence):
	# The first two parameters are the same as the previous function. The data parameter is a matrix or a dataframe that contains the variables that we want to compare to the evidence. The evidence is a dataframe or matrix that has a single row and the same number of columns as data.
	
	p = 1.0
	
	for i in range(evidence.size):
		
		p *= float( np.asarray(np.where( np.logical_and( labels == class_value, data.iloc[:,i] == evidence[i] ) ) ).size ) / float( np.asarray( np.where( labels == class_value ) ).size )
		
	return p
    
def compute_posterior(labels,classes,data,evidence):
	
	priors = compute_priors( labels, classes )
	numerators = np.empty( classes.size , dtype=float )
	
	for entrys in range( priors.size ):
	
		numerators[ entrys ] = priors[ entrys ] * compute_likelihood_data( labels, classes[ entrys ], data, evidence )
	
	return numerators / np.sum( numerators )
	
if __name__ == '__main__':
    main()