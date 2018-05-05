# Implemented cross-validation
# Result of cross-validation
# is an average of correct guess percentages
# for a given number of attempts (as a command-line argument)


# Attributes:
# 1. sepal length in cm		<4.3, 7.9>
# 2. sepal width in cm 		<2.0, 4.4>
# 3. petal length in cm 	<1.0, 6.9>
# 4. petal width in cm 		<0.1, 2.5>

# Classes:
# -- Iris Setosa 
# -- Iris Versicolour 
# -- Iris Virginica

# Source format: attr0,attr1,attr2,attr3,class

from random import shuffle
from functools import reduce
import sys

classes = ["Iris-setosa\n", "Iris-versicolor\n", "Iris-virginica\n"]

# Conversion of values into discrete ranges
discrete_range = range(0,100)
def discrete(value):
	value = float(value)*10
	if value>99:
		return 99
	return int(value)
	

# Takes class probability tables
# and returns class with maximum probability value.
def map_to_class (prob, classes):
	max_prob = prob[classes[0]]
	max_class = classes[0]
	for cl in classes:
		if prob[cl] > max_prob:
			max_prob = prob[cl]
			max_class = cl
	return max_class
	
	
# Takes source data
# then returns class and variable probability lists.
def learn(learning_data):
	
	# Gathering class probabilities
	class_count = {}
	class_prob = {}
	for cl in classes:
		class_count[cl] = 0
	for sample in learning_data:
		class_count[sample[4]] = class_count[sample[4]]+1
	for cl in classes:
		class_prob[cl] = class_count[cl]/len(learning_data)
		
	
	# Gathering variable probabilities.
	
	# How to access:
	# var_prob [var_index] [var_value] [class]
	# it means: probability of a var_index variable
	# having a given value
	# given that object belongs to a given class
	var_prob = [[{cl: 0 for cl in classes} for v in discrete_range] for i in range (0,4)]
	var_count = [[{cl: 0 for cl in classes} for v in discrete_range] for i in range (0,4)]

	
	for sample in learning_data:
		for attr_index in range (0,4):
			var_count[attr_index] [discrete(sample[attr_index])] [sample[4]] = var_count[attr_index] [discrete(sample[attr_index])] [sample[4]] + 1
			
			
	for i in range(0,4):
		for j in discrete_range:
			for cl in classes:
				var_prob[i][j][cl] = (1+var_count[i][j][cl]) / (class_count[cl] + len(learning_data))
				
	return (class_prob, var_prob)
	
	
	
# Takes data
# and classifies samples to given classes.
# For cross-validation purposes it returns percentage of correct guesses.

def classify(data, class_prob, var_prob):
	samples_count = len(data)
	good_predictions = 0
	
	#output = open("results.txt", 'w')
	
	for sample in data:
		prob_predict = {cl: 1 for cl in classes}
		for cl in classes:
			for index in range(0,4):
				prob_predict[cl] = prob_predict[cl] * var_prob[index][discrete(sample[index])] [cl]
			prob_predict[cl] = prob_predict[cl] * class_prob[cl]
			
		predicted_class = map_to_class(prob_predict,classes)
		actual_class = sample[4]	
		#for i in range (0,5):
			#output.write(str(sample[i]) + ", ")
		#output.write("predicted = " + predicted_class + '\n')
		if predicted_class == actual_class:
			good_predictions = good_predictions + 1
			
		percentage = 100*good_predictions/samples_count
		
		
	# print("Good predictions: " + str(good_predictions))
	# print("Number of samples tested: " + str(samples_count))
	# print ("Percentage of good predictions: " + str(percentage) + " %")
	#output.close()
	return percentage



# argv[1]: number of attempts
def main(argv):

	try:
		attempts = int(argv[1])
	except (IndexError, ValueError):
		print("Correct command line format is:\n<script_file_name> <number_of_attempts>")
		sys.exit(1)
	if (attempts<1):
		print("Number of attempts must be positive")
		sys.exit(1)

	# Gathering source data from a given source file
	source_data = []
	source = open("iris.data", "r")
	for line in source:
		if line == "\n":
			break
		temp = line.split(",")
		source_data.append(temp)
	source.close()
	

	# Correct guess percentages list.
	percentages = []
	
	for i in range(0,attempts):
	
		
		# Shuffling for cross-validation purposes.
		shuffle(source_data)
		
		# Dividing between learning and test set in 3:2
		# For 150 elements in iris data, it would be 90:60.
		bound = 90
		learning_data = source_data[0:bound]
		data = source_data[bound:]
		

		(class_prob, var_prob) = learn(learning_data)
		percentage = classify(data, class_prob, var_prob)
		percentages.append(percentage)
		
	# Calculating and printing average correct guess percentage
	result = reduce((lambda x,y: x+y), percentages) / len(percentages)
	print("CROSS-VALIDATION RESULT FOR " + str(attempts) + " ATTEMPTS: "+ str(result) + "%")
	sys.exit()
	
	
if __name__ == "__main__":
	main(sys.argv)