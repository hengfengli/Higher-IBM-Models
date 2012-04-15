# Hengfeng Li
# 383606
# 2012-04-12
"""
Activity Log

### 1st Hour 

I am starting to read the introduction of 4.4 and the whole section of 4.4.1
which presents the IBM model 2 from the book of Statistical Machine 
Translation. I learned that the simple principles of all the IBM models what
they will do and which specific problems they solve. In IBM Model 1, any 
reordering sentences have the same probability, which lacks an alignment 
model. And IBM Model 2 adds the absolute alignment model, which has two steps
of lexical translation step and alignment step. The lexical translation step
is similar with IBM Model 1, but the alignment step needs an alignment 
probability distribution. 

### 2nd Hour

I cannot fully understand the alignment probability distribution, which is 
mainly based on the positions of the input and output words. So I found some
materials to help me understand it. The IBM model 2 adds an alignment 
probability a(a(j)|j, l_e, l_f), in which a(j) is the position of input word,
j is the position of output word, l_e is the length of English sentence, and
l_f is the length of foreign sentence. 

http://www.cs.columbia.edu/~cs4705/notes/ibm12.pdf
www.computing.dcu.ie/~dgroves/CA446/IBMModels.pptx

### 3rd Hour 

I establish a repository for this project on GitHub and begin to read the 
code of Figure 4.7 of EM training algorithm for IBM Model 2 on the book. 
I notice that it needs the code from IBM Model 1 to do a few iterations 
for initializing. Also, I release that I need to look for some sentence 
pairs to train my model so that it can be tested for its correctness. 

### 4th Hour 

I start to implement the EM training algorithm for IBM Model 2. Firstly, 
it needs the code of IBM model 1, so I use my code from worksheet 5. However,
I modify one of the pass-in parameters to the number of iterations. I meet 
some problems about how to initialize an alignment distribution and what 
type should be defined for this structure. Finally, I use a 4-dimension of
default dictionary to represent the alignment probability distribution. 

### 5th Hour

I am confused about the index of j and i in Figure 4.7, which j starts
from 1 to length of English sentence, but i starts from 0 to length of 
foreign sentence. I think all the foreign sentence should begin with a 
"NULL" token. I am looking at the reference code from 
http://code.google.com/p/giza-pp/. I hope I can understand how it does 
the initializing work of alignment distribution. Actually, after reading 
the reference code, I think its function is to translate an English 
sentence to a foreign sentence, which is a little bit different from 
our case. However, I still get some knowledge from the function
"model2::initialize_table_uniformly" in model2.cpp file. 

### 6th Hour

I just simplely add NULL token in the first position of foreign sentences.
I write the code of the initializing part and just one iteration of 
training part. I really carefully check whether I make a mistake on 
the index of j and i. 

### 7th Hour

When I run my code of EM training algorithm, I spend certain time for 
fixing some errors. 

The compiler prompts an type error:

  File "ibm_model2.py", line 165, in EM_training_ibm2
    count_align[i][j][l_e][l_f] += c
TypeError: unsupported operand type(s) for +=: 'type' and 'float'

After I check the content of count_align, I find that I cannot use the 
value of a dictionary for an operand "+=" because I did not defined a 
default value for it. The same error happens for the "total_align". 

### 8th Hour

The "EM_training_ibm2" function can successfully run and return the 
lexical and alignment probability distributions. Originally, I write 
put the English index on the first dimension of the dictionary 
"t_ef" and "count_ef". However, I find that it is not convenient to 
get all possible translations for an foreign word by using that 
structure. So I change the order of dimensions that I put the foreign
index on the first dimension and English index on the second one. 
Although now I can get the lexical and alignment probability distributions,
I still need to think about how to generate a number of different 
translations for a sentence that each has a different probability. 

### 9th Hour 

I am look for materials about how to use lexical and alignment 
probability distribution to find the best translation. 

I find Gawron's slides, which talked about an similar assignment 
to implement IBM model 2. It is more complicated than the algorithm
provided on Koehn's book, because it uses the laplace smoothing 
for the counts of alignment distribution. 

http://www-rohan.sdsu.edu/~gawron/mt_plus/mt/course_core/lectures/assignment_five.pdf

### 10th Hour

I find another project specification from a subject in University
of Amsterdam, which mentions the noisy-channel model. It inspires 
me how to find the best translation for an input sentence.

http://staff.science.uva.nl/~deoskar/ProbGramAndDOP09/projectdescription-SMT.pdf

### 11th Hour 

I go back to read carefully for the 4.3.2 and 4.3.3 sections on Koehn's 
book, which introduces the language model and the noisy-channel model. 

I realize that in noisy-channel model, the translation direction has 
changed from p(e|f) to p(f|e) that is the reason why the reference
code confuses me. I think it is different concept in mathematic, but
in our application, we can just change the input and output sentence.
It means that the input sentence is the English and the output sentence
is the French. For the language model, I have learned N-gram language 
modelling in the lecture 3. 

### 12th Hour

I read carefully for the project specification I found from University of
Amsteradm, it mentions that a fundamental problem in SMT is the word 
aligning a bitext. It said that word aligning a bitext is often the first 
step for training many modern SMT systems. I think it is a reasonable to 
build a word aligner of a bitext based on IBM model 2 at first. 

### 13th Hour

### 14th Hour

### 15th Hour

### 16th Hour

### 17th Hour

### 18th Hour

### 19th Hour

### 20th Hour

### 21th Hour

### 22th Hour

### 23th Hour

### 24th Hour

"""
from __future__ import division
from collections import defaultdict

def EM_training_ibm1(bitexts, num_iter):
	"""
	Return the translation probability model. 

	Arguments:
	bitexts   -- A list contains some sentence pairs. 
	num_iter  -- The number of iterations.

	Returns:
	t_ef         -- A dictionary of translation probabilities. 
	"""
	# Vocabulary of each language
	fr_vocab = set([word for text in bitexts for word in text[0] ])
	en_vocab = set([word for text in bitexts for word in text[1] ])
	# Initial probability
	init_prob = 1 / len(en_vocab)

	# Create the translation model with initial probability
	t_ef = defaultdict(lambda: defaultdict(lambda: init_prob))

	total_e = defaultdict(float)

	# close_to_zero = approx
	# close_to_one  = 1.0 - approx
	# converged = False
	# while not converged:
	for i in range(0, num_iter):
		count_ef = defaultdict(lambda: defaultdict(float))
		total_f = defaultdict(float)

		for fr_set, en_set in bitexts:
			# Compute normalization
			for e in en_set:
				total_e[e] = 0
				for f in fr_set:
					total_e[e] += t_ef[f][e]

			# Collect counts
			for e in en_set:
				for f in fr_set:
					count_ef[f][e] += t_ef[f][e] / total_e[e]
					total_f[f] += t_ef[f][e] / total_e[e]	

		# converged = True
		# Compute the estimate probabilities
		for f in fr_vocab:
			for e in en_vocab:
				t_ef[f][e] = count_ef[f][e] / total_f[f]

				# if t_ef[e][f] >= close_to_zero and t_ef[e][f] <= close_to_one:
				#	converged = False

	return t_ef


def EM_training_ibm2(bitexts, num_iter):

	t_ef = EM_training_ibm1(bitexts, 10)
	
	fr_vocab = set([word for text in bitexts for word in text[0] ])
	en_vocab = set([word for text in bitexts for word in text[1] ])

	align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float))))

	# initialize a(i|j,l_e, l_f) = 1/(l_f + 1)
	for fr_set, en_set in bitexts:
		l_f = len(fr_set) - 1
		l_e = len(en_set)
		initial_value = 1 / (l_f + 1)
		for i in range(0, l_f+1):
			for j in range(1, l_e+1):
				align[i][j][l_e][l_f] = initial_value

	for i in range(0, num_iter):
		count_ef = defaultdict(lambda: defaultdict(float))
		total_f = defaultdict(float)

		count_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
		total_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))

		total_e = defaultdict(float)

		for fr_set, en_set in bitexts:
			l_f = len(fr_set) - 1
			l_e = len(en_set)

			# compute normalization
			for j in range(1, l_e+1):
				en_word = en_set[j-1]
				total_e[en_word] = 0
				for i in range(0, l_f+1):
					total_e[en_word] += t_ef[fr_set[i]][en_word] * align[i][j][l_e][l_f]

			# collect counts
			for j in range(1, l_e+1):
				en_word = en_set[j-1]
				for i in range(0, l_f+1):
					fr_word = fr_set[i]
					c = t_ef[fr_word][en_word] * align[i][j][l_e][l_f] / total_e[en_word]
					count_ef[fr_word][en_word] += c
					total_f[fr_word] += c
					count_align[i][j][l_e][l_f] += c
					total_align[j][l_e][l_f] += c

		# estimate probabilities
		t_ef = defaultdict(lambda: defaultdict(lambda: float))
		align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float))))

		for f in fr_vocab:
			for e in en_vocab:
				t_ef[f][e] = count_ef[f][e] / total_f[f]

		for fr_set, en_set in bitexts:
			l_f = len(fr_set) - 1
			l_e = len(en_set)
			for i in range(0, l_f+1):
				for j in range(1, l_e+1):
					align[i][j][l_e][l_f] = count_align[i][j][l_e][l_f] / total_align[j][l_e][l_f]

	return t_ef, align


bitexts2 = 											\
[ 													\
	(['NULL','das','Haus'], ['the','house']), 		\
	(['NULL','das','Buch'], ['the','book']), 		\
	(['NULL','ein','Buch'], ['a','book']), 			\
	(['NULL','ein','Haus'], ['a', 'house'])			\
]

t_ef, align = EM_training_ibm2(bitexts2, 10)








