from random import shuffle
from functools import reduce
import sys

def learn(learning_data):

	class_count = [0, 0]
	class_prob = [0, 0]
	message_count = 0
	word_counts = [{} for i in range (0,2)]
	word_probs = [{} for i in range (0,2)]
	total_word_count = 0
	
	for i in learning_data:
		message_count = message_count + 1
		for word in i[1].split(" "):
			total_word_count = total_word_count + 1
			class_count[i[0]] = class_count[i[0]]+1
			if word not in word_counts[i[0]]:
				word_counts[i[0]][word] = 1
			else:
				word_counts[i[0]][word] = word_counts[i[0]][word]+1
		
	for i in range (0,2):
		class_prob[i] = class_count[i] / message_count
		for word in word_counts[i].keys():
			# Note: we apply Laplace smoothing in here
			word_probs[i][word] = (1+word_counts[i][word]) / (total_word_count + class_count[i])

	
	return (class_count, class_prob, word_probs, total_word_count)


def classify (source_data, class_count, class_prob, word_probs, total_word_count):
	samples_count = len(source_data)
	good_predictions = 0
	
	# output = open("results.txt", "w")
	
	for sample in source_data:
		prob_predict = [1,1]
		words = sample[1].split(" ")
		
		for i in range (0,2):
			for word in words:
				if word not in word_probs[i].keys():
					# Note: we apply smoothing
					prob_predict[i] = prob_predict[i] * (1/(total_word_count + class_count[i]))
				else:	# Note: word_probs have already applied smoothing
					prob_predict[i] = prob_predict[i] * word_probs[i][word]
					
		predicted_class = prob_predict.index(max(prob_predict))
		if predicted_class == sample[0]:
			good_predictions = good_predictions + 1		
		# output.write("Predicted = " + str(predicted_class) + ", actual = " + str(sample[0])+'\n')
	
	# output.close()
	percentage = good_predictions/samples_count*100
	print("Good predictions: " + str(good_predictions))
	print("Number of samples tested: " + str(samples_count))
	print ("Percentage of good predictions: " + str(percentage) + " %")
	return percentage
	

	
def map_class_string_to_number(string):
	if string == "ham":
		return 1
	return 0
	
	
def main(argv):
	# TODO: Implement Naive Bayes for given data set.
	# Given an SMS, classify if it's ham or spam.
	
	try:
		attempts = int(argv[1])
	except (IndexError, ValueError):
		print("Correct command line format is:\n<script_file_name> <number_of_attempts>")
		sys.exit(1)
	if (attempts<1):
		print("Number of attempts must be positive")
		sys.exit(1)
	
	# Importing messages
	source_data = []
	input_file = open("SMSSpamCollection", encoding="utf-8")
	for i in input_file:
		temp = i.split("\t")
		temp[0] = map_class_string_to_number(temp[0])
		temp[1] = temp[1][0:len(temp[1])-2]	# Cutting off the '\n' char
		source_data.append(temp)
		
	input_file.close()
	
	# Right after that for each message in data:
	# message[0] = ham/spam class, where 0 - spam, 1 - ham
	# message[1] = actual text
	
	bound = int(len(source_data) * 3/5)
	
	# Correct guess percentages list.
	percentages = []
	
	for i in range(0,attempts):
	
		shuffle(source_data)
		
		# Setting bound between learning data and data to be classified
		learning_data = source_data[0:bound]
		data = source_data[bound:]
		
		(class_count, class_prob, word_probs, total_word_count) = learn(learning_data)
		
		percentage = classify(data, class_count, class_prob, word_probs, total_word_count)
		percentages.append(percentage)
		
	# Calculating and printing average correct guess percentage
	result = reduce((lambda x,y: x+y), percentages) / len(percentages)
	print("CROSS-VALIDATION RESULT FOR " + str(attempts) + " ATTEMPTS: "+ str(result) + "%")
	sys.exit()
	

if __name__ == "__main__":
	main(sys.argv)