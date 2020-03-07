# FJTYFilt system

The FJTYFilt system is a series of Python programs for pre-filtering news using an SVM classifier trained by example
(sample files are included).
The system uses the PLOVER data-exchange format: https://github.com/openeventdata/PLOVER; http://ploverdata.org, the spaCy
natural language processing system, and sklearn.

The filter is a simple linear support-vector-machine estimated and implemented using the scikit-learn package and working off word vectors created using the spaCy NLP package. These have been processed using the following utilities

spaCy is used to tokenize the text, then the words are filtered on the following criteria
*length > 3
*all alphabetic
*not part of a named entity: spaCy doesn't always get these correctly
*not in the spaCy English stop list

The remaining word list is transformed into an tf/idf vector by the sklearn function TfidfVectorizer and then a multiclass SVM is estimated using LinearSVC with the default values.

The training cases were developed incrementally using an older corpus using a combination of initially seeding the cases into codeable/not codeable based on whether they had generated events, then, using the program `FJTYFilt_mark_discards.py`, manually classifying about 1000 cases into the various uncodeable categories (again, developed through a couple iterations), and finally "bootstrapping" additional training cases based on classifying unknown cases and then manually reviewing these (which is gets to be quite quick since most of the classifications are correct)

Filter modes
============

The system is configured to use the following scheme:

0. codeable
1. sports
2. culture and entertainment
3. business and finance
4. opinion
5. crime
6. accidents
7. natural disaster
8. [open: current sample uses fisheries]
9. no codeable content: typically links and other non-language records.

A number of training cases have been provided in the file. These produce the following performance

```
SVM_FILTER_ESTIMATE.PY TRAIN/TEST RESULTS
Run datetime: 20-03-07 09:21:56
Training cases proportion: 0.330
Training files
INPUT_DIR: ../Bootstrap-Filter-copy/
  list0215-wordlists.jsonl
  list0218-wordlists.jsonl
  SportOne-wordlists.jsonl
  JunkNine-wordlists.jsonl
  CrimeFive-wordlists.jsonl
  AcciSix-wordlists.jsonl
  WeathrSeven-wordlists.jsonl
  CultTwo-wordlists.jsonl
  OpinFour-wordlists.jsonl

       ============ Experiment 1 ============
Time to estimate: 0.057 sec
Training set
0 |  118     0     0     0     0     0     0     0     0     0    100.00
1 |    0   157     0     0     0     0     0     0     0     0    100.00
2 |    0     0    65     0     0     0     0     0     0     0    100.00
3 |    0     0     0    60     0     0     0     0     3     0    95.24
4 |    1     0     0     0     5     0     0     0     0     0    83.33
5 |    0     0     0     0     0     3     0     0     0     0    100.00
6 |    0     0     0     0     0     0     9     0     0     0    100.00
7 |    0     0     0     0     0     0     0    10     0     0    100.00
8 |    0     0     0     0     0     0     0     0    10     0    100.00
9 |    1     0     0     0     0     0     0     0     0    32    96.97

Time to fit 950 cases 0.126 sec
Test set
              codeable |  233     3     6     8     0     0     0     0     0     1    251 ( 26.42%)   92.83%   92.83%
                sports |    2   264     0     1     0     0     0     0     0     0    267 ( 28.11%)   98.88%   99.25%
 culture/entertainment |   13     7   119     5     0     0     0     0     0     0    144 ( 15.16%)   82.64%   90.97%
      business/finance |   26     3     6    93     0     0     0     0     8     0    136 ( 14.32%)   68.38%   80.88%
               opinion |   10     0     1     2     1     0     0     0     0     0     14 (  1.47%)    7.14%   28.57%
                 crime |   15     1     3     0     0     0     1     0     0     0     20 (  2.11%)    0.00%   25.00%
             accidents |    2     0     2     0     0     0     6     1     0     0     11 (  1.16%)   54.55%   81.82%
      natural disaster |    1     0     0     0     0     0     0    19     0     0     20 (  2.11%)   95.00%   95.00%
                  open |    0     0     0     3     0     0     0     0    13     0     16 (  1.68%)   81.25%  100.00%
   no codeable content |   11     0     0     0     0     0     0     0     0    60     71 (  7.47%)   84.51%   84.51%

```
So, obviously, quite a few additional cases are needed in the `opinion` and `crime` categories, and `business\finance` and `accidents` could use some work as well. These addition cases can be generated using the `FJTYFilt_make_wordlists.py` program




extract_stories.py
Converts all of the files in <news-directory> to .stories format in the file <news-directory> + INFIX + ".stories.txt".  TEST_MODE allows a fixed number of cases (TEST_N) to be generated
mark_discards.py
This utility is used to manually classify cases in a .stories file, simply going through every story in the file and waiting a key-response (no <rtn> needed):
    0-9: enter response and info into the output file
    space: same as '0'
    <enter>: show the full story
    <down-arrow>: skip to next story
    <left-arrow>: exit program
If an output file already exists, there will be an initial query
"Skip previously coded (c) or skip to last (l) file -> " : which has following options
    'c' : skip any cases already in the output file, then append new cases
    'l' : ('el') skip to the frame following the last case that was coded in the output file, then append new cases
    otherwise: start new output file
This has not been tested extensively.


make_labeled_wordlists.py 


make_wordlists_nolabel.py 
Produce unlabelled (first char '-') word vectors from every record in a ".stories.txt"  file. New suffix is ".wordlists.txt" and prefix is "null." : this is used to produce the input file for classify_unlabelled.py


SVM_filter_estimate.py 


Prerequisites
-------------

The current programs are about half-way between research and operational: file names and directories are, for the most part, 
hard-coded in the program, but it would be relatively straightforward to replace this with command-line options (see, for 
example, those implemented in `FILL THIS IN`) so they could be used in a scripted pipeline

Files
=====

All programs are Python 3.7 and open source under the MIT License.

utilFJML.py
------------
Utility routines for the FJOLTYNG-ML system: the only routine used from this is `utilFJML.read_file`, which is a generic
routine for reading *.jsonl* files.

FJTYFilt_make_wordlists.py
----------
Produce word vectors based on the mark\_discards.py output. Program has three command-line options
<classed-files-name> <story-file-name> : names of the  mark\_discards.py output and the corresponding .stories.txt file
<output-file-prefix>: Output file is named <output-file-prefix> + "wordlists.txt" (otherwise default; this must be the final option)
There is a hard-coded list include in the program which can be used to include only categories in that list; the default is to include all categories '0' through '9'.

FJTYFilt_estimator.py
-------------
Estimate and save models using the sklearn modules: input and output formats are hard coded.

TEST_RESULT_FILE_NAME = "SVM_test_results.txt"  (saves a copy of the train/test results)
VECTORZ_PFILE_NAME = "save.vectorizer.p"  (pickled vectorizer)
MODEL_PFILE_NAME = "save.lin_clf.p" (pickled SVM)
The program first does N_EXPERIMENTS (currently set at 5) train/test experiments at a 1:2 ratio (that is, model is estimated on one-third of the cases and tested on the remaining two-thirds): these results are shown on the screen and saved in the file TEST_RESULT_FILE_NAME/ The model which is saved is estimated using all of the cases.

A classification matrix is displayed, followed by these percentages:
category as a percent of all cases
accuracy in classifying the category (main diagonal entry/total)
accuracy in classifying the category as codeable or not (1 - (category-0/total))


FJTYFilt_evaluate.py
--------------------
This program is used to find additional candidate cases from an unclassified set of wordlists. Reads pickled files for a vectorized and model that were generated by SVM_filter_estimate.py then classifies case-word vectors from the file INPUT_FILE_NAME which was generated by make_wordlists_nolabel.py. If the prediction corresponds to MODE, writes the urls of the case to screen and a file OUTPUT_PREFIX + "." + str(MODE) + ".urls.txt". 
The command option -wp writes the wordlists of these predicted cases to a file WORDLIST_PREFIX + "." + str(MODE) + ".wordlists.txt": this is used when these cases will be added to a training set.
The command options -sp and -sf writes the stories of these predicted cases to a file STORY_PREFIX + "." + str(MODE) + ".stories.txt": this is used when manually reviewing the classifications.
TO RUN PROGRAM:
python3 classify_unlabelled.py -m <mode> [optional command pairs]
Command option occur in pairs -<option> <value>. -m mode is required

    -wf INPUT_FILE_NAME : name of the wordlist file of unlabelled vectors to be classified. Default: hard-coded name in program
   -fp OUTPUT_PREFIX   : prefix for the file which lists of the urls that were predicted as being MODE. Default: "Mode"
   -sp STORY_PREFIX    : prefix for file of stories for the cases that were predicted as being MODE. Default: do not write file
   -sf STORY_FILE_NAME : name of .stories.txt file used to generate the unlabelled vectors. Required if -sp is used
   -wp WORDLIST_PREFIX : prefix for file of wordlists for the cases that were predicted as being MODE. Default: do not write file
Other utilities
pattern_SVM_encode.py 
Produces labelled word vectors based on a regex pattern and spaCy processing


