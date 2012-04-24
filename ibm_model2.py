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

There is an implementation of word aligner in nltk packge, but it is based
on IBM Model 1. So I can use the class of Alignment and AligneSent to 
implement a word aligner based on IBM Model 2. 

https://github.com/nltk/nltk/blob/master/nltk/align.py

I find a little bug of IBMModel 1 on the '2.0.1rc3' version of nltk. When
I use the aligned function in IBMModel1, it prompts 

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Library/Python/2.7/site-packages/nltk-2.0.1rc3-py2.7.egg/nltk/align.py", line 396, in aligned
    if self.probablities is None:
AttributeError: 'IBMModel1' object has no attribute 'probablities'

I think there is a spelling mistake on that version.

### 14th Hour

I create the IBMModel2 class, make a train method, change all the previous
code into that class. Also, I need to test its correctness after moving. 

Besides, I read another material about word alignment models, which did 
the similar stuff as our case. And it mentions the requirements about 
the implementation of a word aligner for Machine Translation. 

http://www.stanford.edu/class/cs224n/pa/pa2.pdf

### 15th Hour

I spend some time to use the AlignedSent class to replace my original 
'bitexts', which can take advantage of Alignment class. Besides, there is
a aligned corpus 'comtrans' in nltk packge, which uses AlignedSent to 
represent its items. Also, I do some tests to ensure the correctness 
for my program.

### 16th Hour

In IBM model 2, there are lexical and alignment probabilities, but the 
problem is how to use them to find out the best word alignment. 

The aligned function of IBMModel1 in nltk packge finds the maximum t(e|f)
for each word in English sentence and stores all links on an instance of 
class Alignment. 

My simple thought is to use the a(i|j,l_e,l_f). With given j, length of 
English sentence and length of French sentence, we can find which i has 
the maximum probability. 

I change the index of 't_ef' and 'count_ef' back to the original one, 
which put English word at first and foreign word at second because 
I realize that this is more convenient to understand the code.  

### 17th Hour

After I read the formula 4.25 on section 4.4.1 of textbook, I think 
IBMModel2 should be based on the same way from IBMModel1. However, 
it should time with the corresponding alignment probability in 
IBM Model 2. The material from following link gives me confidence 
to complete my thought.

http://www.cs.columbia.edu/~cs4705/hw2/h2-programming.pdf

### 18th Hour

When I run my program, the compiler prompts an error:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "ibm_model2.py", line 246, in align
    max_alignProb = (self.probabilities[en_word][None]
                     *self.alignments[0][j][l_e][l_f], None)
TypeError: unsupported operand type(s) for *: 'float' and 'type'

I think it is the initializing problem of alignments distribution
probabilities. When initializing the defaultdict structure, I just 
set a data type for it, but I didn't set a default value for it. 
That is the reason why the compiler says the operand cannot be used
on 'float' and 'type'.

After I fix the above problem, the program can successfully run. However,
when I test it with reordering sentence, all the English word are linked
with Null token. After I check the content of lexical and alignment 
distribution probabilities, I find that the alignment probabilities is 
zero even when lexical probability has a higher value. So I think it needs
to do the laplace smoothing, which add 0.5 for probability weighted counts.

This smoothing method is reference from the following link:
http://www-rohan.sdsu.edu/~gawron/mt_plus/mt/course_core/lectures/assignment_five.pdf

### 19th Hour

Now, I can use the aligned sentence of comtrans corpus in nltk package to 
do some tests to validate the correctness of my program. My evaluation is 
very simple that after training the model, using the program finds the 
alignments for the training data so that its precision can be tested. 
I notice that the AlignedSent class provides a 'precision' method to 
compare two alignment. So I have the correct alignment that can be the 
golden standard. 

### 20th Hour

I use the first 100 sentences in comtrans corpus to train my model, 
which spends 10 seconds and average 140MB memory. 

I test 100 sentences to 300 sentences by using my model.

=======================================================
sents  |  en num  |  fr num  |   time    |   memory
-------------------------------------------------------
100        616        666       10sec         140MB
200       1011       1143       26sec         200MB
300       1411       1640       69sec         470MB
=======================================================

Also, I make a comparison between the IBMModel1 in nltk package 
and the IBMModel1 I written.

=======================================================
sents |     nltk version   ||      my version     
      |   time   |  memory ||    time    |   memory
-------------------------------------------------------
100      32.5sec     190MB      5.7sec         130MB
200     103.5sec     400MB     14.5sec         190MB
300     260.2sec     750MB     35.8sec         490MB
=======================================================

From above table, I think the time does not make any 
sense, because my model 1 only runs for 10 iterations 
while the nltk version runs to a converged state.

However, it is interested in the memory comparisons that
nltk model 1 uses 750MB memory when training for 300 sentences, 
which is larger than my version. 

I check the source code of IBMModel1 class and I find the obvious
difference is the representation of multi-dimensional dictionary. 
It uses defaultdict[a,b,c,d], but I uses the way of defaultdict[a][b][c][d].
I make a test that changing all my representation to defaultdict[a,b,c,d].
And I find that my model 1 also needs more than 750MB when train 
300 sentences, which proves my thought that defaultdict[a][b][c][d]
is using less memory than defaultdict[a,b,c,d] in Python. 

### 21th Hour

Now, my problem is that the model does not have a good precision 
with training for small sentences, but the large sentences takes
a huge memory space, which is impossible for normal computer. 

After I make some checks on my program, spending the huge memory
happens on the following code:

for f in fr_vocab:
    for e in en_vocab:
        t_ef[e][f] = count_ef[e][f] / total_f[f]

I cannot figure out the reason why it takes a huge memory. For 
example, in terms of 300 sentences, there are 1411 English words
and 1640 foreign words and each float64 type takes 8 Bytes, so 
it should take 2.3M * 8 Bytes = 18.4 MB. 

Even when I change the code like following, just assign a float 
value to it on my model 1. 

for f in fr_vocab:
    for e in en_vocab:
        t_ef[e][f] = 0.1

It also takes 470MB memory when running. 

### 22th Hour

The problem is how the model is able to be trained a large corpus 
to improve its precision.

After getting the advice to use an unique integer to represent each word,
I start to have a try on this approach. Instead of using the corpus directly,
I made a pre-processing on the corpus so that each word is recorded by an 
unique number on a dictionary. And then, each word is replaced by that 
unique number in the sentence.

However, after I test its memory usage, this approach does not reduce 
the memory used by training model. I think the reason is that in a 
dictionary, the hash value of a string or an integer takes the same
memory space. 

### 23th Hour

I find that the Alignment class not only has the function for computing 
precision, but also there are functions for computing recall and 
error rate of alignment. So I use these functions to provide an 
evaluation on the alignments generated by the model.

I make a function 'alignSents' which is able to align multiple sentences,
and return the evaluations. 

### 24th Hour

I write more comments to explain the class and functions. Also, I make some 
doctest to test the functions and clearly explain how the function can be 
used.

I think the way of increasing the size of training data in Python is to 
use C language writing some modules in order to improving the performnce
and reducing the usage of memory space. However, this project requires 
pure Python implementation so we can try it on next project. 

"""
from __future__  import division
from collections import defaultdict
from nltk.align  import AlignedSent
from nltk.align  import Alignment
from nltk.corpus import comtrans


def replaceByNumber(bitexts):
    """
    This function implements the approach to replace a word by an 
    number in the sentence pairs. 

    >>> en_dict, fr_dict, bitexts = replaceByNumber(comtrans.aligned_sents()[:100])
    >>> en_dict['Kommission']
    474
    >>> fr_dict['check']
    331
    >>> bitexts[0]
    AlignedSent([1, 2, 3], [1, 2, 3, 4], Alignment([(0, 0), (1, 1), (1, 2), (2, 3)]))

    Arguments:
    bitexts   -- A list of instances of AlignedSent class, which 
                 contains sentence pairs. 

    Returns:
    en_dict         -- A dictionary with an English word as a key and
                       an integer as a value.
    fr_dict         -- A dictionary with an foreign word as a key and
                       an integer as a value.
    new_bitexts     -- A list of instances of AlignedSent class, which
                       the sentence pairs that each word is represented 
                       by a number.
    """
    new_bitexts = []

    # Assign zero as an initial value
    en_dict = defaultdict(lambda: 0)
    fr_dict = defaultdict(lambda: 0)

    # The number starts from one to represent each word
    en_count = 1
    fr_count = 1
    for aligned_sent in bitexts:
        new_words = []
        for word in aligned_sent.words:
            if en_dict[word] == 0:
                en_dict[word] = en_count
                en_count += 1
            # Append the integer to the new sentence
            new_words.append(en_dict[word])

        new_mots = []
        for mots in aligned_sent.mots:
            if fr_dict[mots] == 0:
                fr_dict[mots] = fr_count
                fr_count += 1
            # Append the integer to the new sentence
            new_mots.append(fr_dict[mots])

        # Create a new instance of AlignedSent class 
        # and append it to new list of sentence pairs.
        new_bitexts.append(AlignedSent(new_words, new_mots, 
                                       aligned_sent.alignment))

    return en_dict, fr_dict, new_bitexts

class IBMModel2(object):
    """
    This class implements the algorithm of Expectation Maximization for 
    the IBM Model 2. 

    Step 1 - Run a number of iterations of IBM Model 1 and get the initial
             distribution of translation probability. 

    Step 2 - Collect the evidence of a English word being translated by a 
             foreign language word.

    Step 3 - Estimate the probability of translation and alignment according 
             the evidence from Step 2. 

    >>> bitexts = comtrans.aligned_sents()[:100]
    >>> ibm = IBMModel2(bitexts, 5)
    >>> aligned_sents = ibm.alignSents(bitexts)
    >>> aligned_sents[0].words
    ['Wiederaufnahme', 'der', 'Sitzungsperiode']
    >>> aligned_sents[0].mots
    ['Resumption', 'of', 'the', 'session']
    >>> aligned_sents[0].alignment
    Alignment([(0, 0), (1, 2), (2, 3)])
    >>> bitexts[0].precision(aligned_sents[0])
    0.75
    >>> bitexts[0].recall(aligned_sents[0])
    1.0
    >>> bitexts[0].alignment_error_rate(aligned_sents[0])
    0.1428571428571429

    >>> prec, recall, error_rate = ibm.evaluate(bitexts, aligned_sents)
    >>> prec
    0.35018511941714076
    >>> recall
    0.39611872732188913
    >>> error_rate
    0.6305001263145995

    """
    def __init__(self, alignSents, num_iter):
        self.num_iter = num_iter
        self.probabilities, self.alignments = self.EM_training_ibm2(alignSents)

    def EM_training_ibm1(self, alignSents, num_iter):
        """
        Return the translation probability model trained by IBM model 1. 

        Arguments:
        alignSents   -- A list of instances of AlignedSent class, which 
                        contains sentence pairs. 
        num_iter     -- The number of iterations.

        Returns:
        t_ef         -- A dictionary of translation probabilities. 
        """

        # Vocabulary of each language
        fr_vocab = set()
        en_vocab = set()
        for alignSent in alignSents:
            en_vocab.update(alignSent.words)
            fr_vocab.update(alignSent.mots)
        # Add the Null token
        fr_vocab.add(None)

        # Initial probability
        init_prob = 1 / len(en_vocab)

        # Create the translation model with initial probability
        t_ef = defaultdict(lambda: defaultdict(lambda: init_prob))

        total_e = defaultdict(lambda: 0.0)

        for i in range(0, num_iter):
            count_ef = defaultdict(lambda: defaultdict(lambda: 0.0))
            total_f = defaultdict(lambda: 0.0)

            for alignSent in alignSents:
                en_set = alignSent.words
                fr_set = [None] + alignSent.mots  

                # Compute normalization
                for e in en_set:
                    total_e[e] = 0.0
                    for f in fr_set:
                        total_e[e] += t_ef[e][f]

                # Collect counts
                for e in en_set:
                    for f in fr_set:
                        c = t_ef[e][f] / total_e[e]
                        count_ef[e][f] += c
                        total_f[f] += c

            # Compute the estimate probabilities
            for f in fr_vocab:
                for e in en_vocab:
                    t_ef[e][f] = count_ef[e][f] / total_f[f]

        return t_ef     

    def EM_training_ibm2(self, alignSents):
        """
        Return the translation and alignment probability distributions
        trained by the Expectation Maximization algorithm for IBM Model 2. 

        Arguments:
        alignSents   -- A list contains some sentence pairs. 
        num_iter     -- The number of iterations.

        Returns:
        t_ef         -- A distribution of translation probabilities.
        align        -- A distribution of alignment probabilities.
        """

        # Get initial translation probability distribution
        # from a few iterations of Model 1 training.
        t_ef = self.EM_training_ibm1(alignSents, 10)

        # Vocabulary of each language
        fr_vocab = set()
        en_vocab = set()
        for alignSent in alignSents:
            en_vocab.update(alignSent.words)
            fr_vocab.update(alignSent.mots)
        fr_vocab.add(None)

        align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float))))

        # Initialize the distribution of alignment probability,
        # a(i|j,l_e, l_f) = 1/(l_f + 1)
        for alignSent in alignSents:
            en_set = alignSent.words
            fr_set = [None] + alignSent.mots
            l_f = len(fr_set) - 1
            l_e = len(en_set)
            initial_value = 1 / (l_f + 1)
            for i in range(0, l_f+1):
                for j in range(1, l_e+1):
                    align[i][j][l_e][l_f] = initial_value


        for i in range(0, self.num_iter):
            count_ef = defaultdict(lambda: defaultdict(float))
            total_f = defaultdict(float)

            count_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            total_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))

            total_e = defaultdict(float)

            # for fr_set, en_set in bitexts:
            for alignSent in alignSents:
                en_set = alignSent.words
                fr_set = [None] + alignSent.mots
                l_f = len(fr_set) - 1
                l_e = len(en_set)

                # compute normalization
                for j in range(1, l_e+1):
                    en_word = en_set[j-1]
                    total_e[en_word] = 0
                    for i in range(0, l_f+1):
                        total_e[en_word] += t_ef[en_word][fr_set[i]] * align[i][j][l_e][l_f]

                # collect counts
                for j in range(1, l_e+1):
                    en_word = en_set[j-1]
                    for i in range(0, l_f+1):
                        fr_word = fr_set[i]
                        c = t_ef[en_word][fr_word] * align[i][j][l_e][l_f] / total_e[en_word]
                        count_ef[en_word][fr_word] += c
                        total_f[fr_word] += c
                        count_align[i][j][l_e][l_f] += c
                        total_align[j][l_e][l_f] += c

            # estimate probabilities
            t_ef = defaultdict(lambda: defaultdict(lambda: 0.0))
            align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))

            # Smoothing the counts for alignments
            for alignSent in alignSents:
                en_set = alignSent.words
                fr_set = [None] + alignSent.mots
                l_f = len(fr_set) - 1
                l_e = len(en_set)

                laplace = 1.0
                for i in range(0, l_f+1):
                    for j in range(1, l_e+1):
                        value = count_align[i][j][l_e][l_f]
                        if value > 0 and value < laplace:
                            laplace = value

                laplace *= 0.5 
                for i in range(0, l_f+1):
                    for j in range(1, l_e+1):
                        # print i,j,l_e,l_f
                        count_align[i][j][l_e][l_f] += laplace

                initial_value = laplace * l_e
                for j in range(1, l_e+1):
                    total_align[j][l_e][l_f] += initial_value
            
            # Estimate the new lexical translation probabilities
            for f in fr_vocab:
                for e in en_vocab:
                    t_ef[e][f] = count_ef[e][f] / total_f[f]

            # Estimate the new alignment probabilities
            for alignSent in alignSents:
                en_set = alignSent.words
                fr_set = [None] + alignSent.mots
                l_f = len(fr_set) - 1
                l_e = len(en_set)
                for i in range(0, l_f+1):
                    for j in range(1, l_e+1):
                        align[i][j][l_e][l_f] = count_align[i][j][l_e][l_f] / total_align[j][l_e][l_f]

        return t_ef, align

    def align(self, alignSent):
        """
        Returns the alignment result for one sentence pair. 
        """

        if self.probabilities is None or self.alignments is None:
            raise ValueError("The model does not train.")

        alignment = []

        l_e = len(alignSent.words);
        l_f = len(alignSent.mots);

        for j, en_word in enumerate(alignSent.words):
            
            # Initialize the maximum probability with Null token
            max_alignProb = (self.probabilities[en_word][None]*self.alignments[0][j+1][l_e][l_f], None)
            for i, fr_word in enumerate(alignSent.mots):
                # Find out the maximum probability
                max_alignProb = max(max_alignProb,
                    (self.probabilities[en_word][fr_word]*self.alignments[i+1][j+1][l_e][l_f], i))

            # If the maximum probability is not Null token,
            # then append it to the alignment. 
            if max_alignProb[1] is not None:
                alignment.append((j, max_alignProb[1]))

        return AlignedSent(alignSent.words, alignSent.mots, alignment)

    def alignSents(self, alignSents):
        """
        Returns the alignment result for several sentence pairs. 
        """

        if self.probabilities is None or self.alignments is None:
            raise ValueError("The model does not train.")

        aligned_sents = []
        for sent in alignSents:
            new_alignSent = AlignedSent(sent.words, sent.mots)
            aligned_sents.append(self.align(new_alignSent))

        return aligned_sents

    def evaluate(self, bitexts, alignSents):
        """
        Returns the evaluation for the alignments of several 
        sentence pairs.
        """
        count = 0 
        prec   = []
        recall = []
        error_rate = []
        for item in alignSents:
            prec.append(bitexts[count].precision(item))
            recall.append(bitexts[count].recall(item))
            error_rate.append(bitexts[count].alignment_error_rate(item))
            count += 1
        return sum(prec)/count, sum(recall)/count, sum(error_rate)/count

# run doctests
if __name__ == "__main__":
    import doctest
    doctest.testmod()

