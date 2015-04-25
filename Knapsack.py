#In class knapsack problem example
from random import randint
import numpy as np

class Knapsack:
	def __init__(self):
		self.V = 100
		self.L = 5
		self.v = [10, 2, 3, 100, 50]
		self.p = [4,10,40,5,75]
		
	def fitness(self, s):
	
		total = 0
		total_volume = 0
		for i in range(len(s)):
			if s[i] == 1:
				total += self.p[i]
				total_volume += self.v[i]
				
			if total_volume > self.V:
				total -= 2*self.p[i]
	
		return total
	
	def tournament(self, population):
		offspring = []
		nTournament = 4
		for i in range(1):
			parents = []
			for j in range(nTournament):
				parents.append(randint(0, len(population) - 1))
			fitness_values = []
			for j in range(nTournament):
				fitness_values.append( self.fitness(population[parents[j]]) )
			a = -1*np.array(fitness_values)
			#print fitness_values
			#print a
			inxs = a.ravel().argsort()
			#print inxs
			p1 = inxs[0]
			p2 = inxs[1]
			#print population[p1], population[p2]
			crossoverpoint = randint(0, self.L-2)
			offspring1 = population[p1][0:(crossoverpoint+1)] + population[p2][(crossoverpoint+1)]
			offspring2 = population[p2][0:(crossoverpoint+1)] + population[p1][(crossoverpoint+1)]
			print offspring1, offspring2

	
knapsack = Knapsack()
population = [[1,1,1,1,1], [1,0,0,1,1],[0,0,0,1,1],[1,0,1,1,1],[0,1,0,1,0],[1,1,1,1,1], [1,0,0,1,1],[0,0,0,1,1],[1,0,1,1,1],[0,1,0,1,0]]

for solution in population:
	print(knapsack.fitness(solution))
knapsack.tournament(population)