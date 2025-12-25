"""
15-110 Hw6 - Language Modeling Project
Name: Adrian Lee
AndrewID: adrianle
"""

import hw6_language_tests as test

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename): #open file, split function prob needed
    list1 = [] 
    f = open(filename, "r")
    for text in f:
            line = text.strip().split() #splits line by space
            if line:
                list1.append(line) #else statement needed?
    return list1 #f.close doesn't work-> future issue?

'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
corpus = [ ["hello", "world"],
["hello", "world", "again"] ]

def getCorpusLength(corpus): #len function
    words = 0
    for unigram in corpus:
        words += len(unigram)
    return words


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus): #iterate and index-> will probably use append function
    list2 = []
    for new in corpus:
        for old in new: #double for loop so it iterates through 2D list in corpus
            if old not in list2: 
                list2.append(old)
    return list2

'''
makeStartCorpus(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: 2D list of strs
'''
def makeStartCorpus(corpus): #indexing for this one
    list3 = []
    for new in corpus:
        if len(new) > 0:
            list3.append([new[0]])#list3.append([new2[0]])
            #list3.append([new2[1][0]])-> Why this doesn't work?
    return list3

'''
countUnigrams(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus): #dictionary, checks length of corpus and maybe a double loop?
    list4 = {}
    for variable in corpus:
        for variable2 in variable: #double for loop ensures that it will loop through each word
            if variable2 in list4:
                list4[variable2] = list4[variable2] + 1 #will add count by 1
            else:
                list4[variable2] = 1 #if the variable is not in the list, it adds it as a new word with a count of 1
    return list4


'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    name = {} #step 1
    for sentence in corpus:
        for i in range(len(sentence)- 1): #step 2
            variablename= (sentence[i], sentence[i + 1]) #step 3
            if variablename[0] not in name:
                name[variablename[0]] = {} #step 4
            if variablename[1] not in name[variablename[0]]: #step 5
                name[variablename[0]][variablename[1]] = 1
            else:
                name[variablename[0]][variablename[1]] += 1
    return name


'''
separateWords(line)
#7 [Check6-1]
Parameters: str
Returns: list of strs
'''
import string

def separateWords(line):
    list4=[]
    nospace = ""
    for hairband in line:
        if hairband in string.punctuation and hairband != "'":
            if nospace:
                list4.append(nospace)
            list4.append(hairband)
            nospace = ""
        elif hairband == " ":
            if nospace:
                list4.append(nospace)
                nospace = ""
        else:
            nospace += hairband
    if nospace:
        list4.append(nospace)
    return list4

'''
cleanBookData(text)
#8 [Check6-1]
Parameters: str
Returns: str
'''

def cleanBookData(text):
    insert=[]
    for anything in text.splitlines():
        finish=[]
        function = separateWords(anything)
        for anything2 in function:
            finish.append(anything2)
            if anything2[-1] in ".!?":
                insert.append(" ".join(finish))
                finish=[]
        if finish:
            insert.append(" ".join(finish))
    return "\n".join(insert).lower()


### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    prob = 1/len(unigrams)
    start = []
    for anything in unigrams:
        start.append(prob)
    return start 


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    empty = []
    for anything in unigrams:
        if anything in unigramCounts:
            count = unigramCounts[anything]
        else:
            count = 0
        divide = count / totalCount
        empty.append(divide)
    return empty


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    step1 = {} #new dictionary
    for prevWord in bigramCounts: #iteratres through each word (step 2)
        keys = [] #one for the words (keys) in bigramCounts
        prob = [] #one for the probabiities of those words
        for variable in bigramCounts[prevWord]: #step 2c
            keys.append(variable)
            prob.append(bigramCounts[prevWord][variable] / unigramCounts[prevWord]) #note 1
        temp = {"words": keys, "probs": prob} #step 2d
        step1[prevWord] = temp #step 2e
    return step1 #step 3

'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''


def getTopWords(count, words, probs, ignoreList):
    dictionary ={}
    search = []
    prob = []
    for i in range(len(words)):
        if words[i] not in ignoreList:
            search.append(words[i])
            prob.append(probs[i])
    while len(dictionary) < count:
        index = prob.index(max(prob))
        if search[index] not in dictionary:
            dictionary[search[index]] = prob[index]
        search.pop(index)
        prob.pop(index)
    return dictionary


'''def getTopWords(count, words, probs, ignoreList):
    highprob = {}
    while len(highprob) < count:
        maxProb = max(probs)
        index = probs.index(maxProb)
        if words[index] not in ignoreList and words[index] not in highprob:
            highprob[words[index]] = maxProb
        probs.pop(index)
        words.pop(index)
    return highprob''' ##This works within Thonny, but not in the testcases within Gradescope 


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices

def generateTextFromUnigrams(count, words, probs):
    string = ""
    for i in range(count): #
        function = choices(words, weights=probs)[0] #indexes first element of that list
        string = string + function + " " #or the spaces
    return string.strip()

'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
from random import choices

def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    string = []
    for i in range(count):
        if not string or string[-1] in ".!?": #nothing has been added or last word added was punctuation
            function1 = choices(startWords, weights=startWordProbs) #choices function 
            string.append(function1[0]) #appends to string first word 
        else:
            function2 = choices(bigramProbs[string[-1]]["words"], weights=bigramProbs[string[-1]]["probs"]) 
            string.append(function2[0])
    last = " ".join(string) #adds spaces
    return last

 

### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    length = getCorpusLength(corpus) #Check6-1 Q2
    vocab = buildVocabulary(corpus) #Check6-1 Q3
    unigrams = countUnigrams(corpus) #Check6-1 Q5
    unigramprob = buildUnigramProbs(vocab, unigrams, length) #Check6-1 Q1
    common = getTopWords(50, vocab, unigramprob, ignore)
    barPlot(common, "Probability of the top 50 Most Common Words")
    return


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    computeStartWord = makeStartCorpus(corpus)
    unique = buildVocabulary(computeStartWord) 
    count = countUnigrams(computeStartWord)
    length = getCorpusLength(computeStartWord)
    prob = buildUnigramProbs(unique, count, length)
    step4 = getTopWords(50, unique, prob, ignore)
    barPlot(step4, "Top Starting Words")
    return


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    forunigrams = countUnigrams(corpus) #need both unigram counts and bigram counts as the arguments for buildBigramProbs
    forbigrams = countBigrams(corpus)
    function1 = buildBigramProbs(forunigrams, forbigrams) 
    x = function1[word]["words"]
    y = function1[word]["probs"]
    function2 = getTopWords(10, x, y, ignore)
    barPlot(function2, "Top 10 Next Words")    
    return


'''
setupData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupData(corpus1, corpus2, topWordCount):
#previousfunctions
    vocabforcorp1 = buildVocabulary(corpus1)
    vocabforcorp2 = buildVocabulary(corpus2)
    uniforcorp1 = countUnigrams(corpus1)
    uniforcorp2 = countUnigrams(corpus2)
    length1 = getCorpusLength(corpus1)
    length2 = getCorpusLength(corpus2)
    unigramforcorp1 = buildUnigramProbs(vocabforcorp1, uniforcorp1, length1)
    unigramforcorp2 = buildUnigramProbs(vocabforcorp2, uniforcorp2, length2)
    function1 = getTopWords(topWordCount, vocabforcorp1, unigramforcorp1, ignore)
    function2 = getTopWords(topWordCount, vocabforcorp2, unigramforcorp2, ignore)

    everything = list(function1.keys())
    for word in function2.keys():
        if word not in everything:
            everything.append(word)
    prob1 = []
    prob2 = []
    for what in everything:
        if what in unigramforcorp1:
            prob1.append(unigramforcorp1[what])
        else:
            prob1.append(0)
        if what in unigramforcorp2:
            prob2.append(unigramforcorp2[what])
        else:
            prob2.append(0)

    dictionary = {
        "Top Words": everything,
        "Corpus 1 Probability": prob1,
        "Corpus 2 Probability": prob2
    }
    return dictionary

'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    chart1 = setupChartData(corpus1,corpus2,numWords)
    sideBySideBarPlots(chart1["topWords"],chart1["corpus1Prob"],chart1["corpus1Prob"], name1, name2, title)
    return


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    chart2 = setupChartData(corpus1,corpus2,numWords)
    scatterPlot(chart2["corpus1Prob"], chart2["corpus2Prob"],chart2["topWords"],title)
    return
##assert topWords in result 

### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()


    ## Uncomment these for Week 3 ##

    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
