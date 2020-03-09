# FJTYFilt system

The FJTYFilt system is a series of Python programs for pre-filtering news using an SVM classifier trained by example
(sample files are included).
The system uses the PLOVER data-exchange format: https://github.com/openeventdata/PLOVER; http://ploverdata.org, the spaCy
natural language processing system, and sklearn.

The filter is a simple linear support-vector-machine estimated and implemented using the [scikit-learn package](https://scikit-learn.org/stable/modules/svm.html) and working off word vectors created using the [spaCy NLP package](https://spacy.io/). 

`spaCy` is used to tokenize the text, then the words are filtered on the following criteria

* length > 3
* all alphabetic
* not part of a named entity: spaCy doesn't always get these correctly
* not in the spaCy English stop list

The remaining word list is transformed into an tf/idf vector by the sklearn function `TfidfVectorizer` and then a multiclass SVM is estimated using `LinearSVC` with the default values.

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

A number of training cases have been provided in the file `FJTY_training_wordlists.zip`. These produce the following performance

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


Prerequisites
=============

1. `spaCy` and `sklearn` need to be installed.

Files
=====

All programs are Python 3.7 and open source under the MIT License.

utilFJML.py
------------
Utility routines for the FJOLTYNG-ML system: the only routine used from this is `utilFJML.read_file`, which is a generic
routine for reading *.jsonl* files.

mark_discards.py
----------------
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

FJTYFilt_make_wordlists.py
--------------------------
Produce word vectors based on the mark_discards.py output. Program has three command-line options
<classed-files-name> <story-file-name> : names of the  mark_discards.py output and the corresponding .stories.txt file
<output-file-prefix>: Output file is named <output-file-prefix> + "wordlists.txt" (otherwise default; this must be the final option)
There is a hard-coded list include in the program which can be used to include only categories in that list; the default is to include all categories '0' through '9'.

FJTYFilt_estimator.py
---------------------
Estimate and save models using the `sklearn` modules: input and output formats are hard coded.

* TEST_RESULT_FILE_NAME = "SVM_test_results.txt"  (saves a copy of the train/test results)
* VECTORZ_PFILE_NAME = "save.vectorizer-Mk2.p"  (pickled vectorizer)
* MODEL_PFILE_NAME = "save.lin_clf-Mk2.p" (pickled SVM)

The program first does N_EXPERIMENTS (currently set at 5) train/test experiments at a 1:2 ratio (that is, model is estimated on one-third of the cases and tested on the remaining two-thirds): these results are shown on the screen and saved in the file TEST_RESULT_FILE_NAME/ The model which is saved is estimated using all of the cases.

A classification matrix is displayed, followed by these percentages:
category as a percent of all cases
accuracy in classifying the category (main diagonal entry/total)
accuracy in classifying the category as codeable or not (1 - (category-0/total))


FJTYFilt_evaluate.py
--------------------
This program classifies new cases: it reads pickled files for a vectorized and model that were generated by `FJTYFilt_estimator.py` then classifies case-word vectors from the file `INPUT_FILE_NAME` which was generated by `FJTYFilt_make_wordlists.py`. If the prediction corresponds to MODE, writes the urls of the case to screen and a file `OUTPUT_PREFIX + "." + str(MODE) + ".urls.txt"`. A later program is then used to merge these based on the `id` field.

TO RUN PROGRAM: `python3 FJTYFilt_evaluate.py -m <mode> [optional command pairs]`

Example for filtering sport stories:
```
FJTYFilt_evaluate.py m -1
```
Command options occur in pairs -<option> <value>. -m mode is required

* -wf INPUT_FILE_NAME : name of the wordlist file of unlabelled vectors to be classified. Default: hard-coded name in program
* -fp OUTPUT_PREFIX   : prefix for the file which lists of the urls that were predicted as being MODE. Default: "Mode"
* -sp STORY_PREFIX    : prefix for file of stories for the cases that were predicted as being MODE: this is used when manually reviewing the classifications. Default: do not write file
* -sf STORY_FILE_NAME : name of .stories.txt file used to generate the unlabelled vectors. Required if -sp is used
* -wp WORDLIST_PREFIX : prefix for file of wordlists for the cases that were predicted as being MODE:  this is used when these cases will be added to a training set. Default: do not write file

`FJTYFilt_evaluate.py` is currently configured to evaluate only a single discard category at a time; for operational use in a pipeline it makes more sense to evaluate all of the possibilities, and the code can be easily modified to handle this.

Supporting files
================

FJTY_training_wordlists.zip
---------------------------
Set of training cases: most of these are mode-specific; the remaining two are a mixed set produced using `mark\_discards.py`


FJTY_SVM_Models.zip
--------------------
Estimated SVM models: `FJTYFilt_evaluate.py` is currently set to use these


demo-REUT-20-02-25-wordlists.jsonl
----------------------------------
Sample input for `FJTYFilt_evaluate.py`

*prodigy* and other classification utilities
===================

An article on machine learning in ]*The Economist*](https://www.economist.com/technology-quarterly/2020/01/02/chinas-success-at-ai-has-relied-on-good-data) in late 2019 made the interesting observation that China's success in this area rests
not on new algorithms---they use the same open source tools everyone else uses---but on their ability to quickly and inexpensively 
generate very large numbers of labelled training cases: an entire industry has arisen in China to do this.

This is also the insight behind the [explosion.ai](https://explosion.ai/) program [`prodigy`](https://prodi.gy/): enable a user or small team to rapidly 
label/classify training cases. `prodigy` is proprietary software but explosion.ai has certainly made far more than their share of contributions 
to open source---`spaCy` for godsakes!---and this is a place where the investment might be well worthwhile. The key contribution of
`prodigy` is the integration of a machine-learning algorithm which, well, "learns" the correct classification of your cases, so pretty
soon you are simply approving its decisions rather than having to think: this allows classification to go *very* fast.

The `NAME` program below is an alternative way of doing this---it has a very small footprint, is keyboard based, and works well on 
airplanes---but does not have the machine learning component. If you are going to be doing a lot of labelling, notably in the development
of new categories, I'd recommend `prodigy`. The two programs  `name` and `name` convert between the PLOVER and `prodigy` formats.




Additional notes
================

1. The current programs are about half-way between research and operational: file names and directories are, for the most part, 
hard-coded in the program, but it would be relatively straightforward to replace this with command-line options (see, for 
example, those implemented in `FJTYFilt_evaluate.py`) so they could be used in a scripted pipeline

2. MAKE A NOTE on configuring a pipeline to avoid running spaCy twice.

