# FJTYFilt system

The FJTYFilt system is a series of Python programs for pre-filtering news articles using an SVM classifier trained by example
(sample files are included).
The system uses the [PLOVER data-exchange format](https://github.com/openeventdata/PLOVER): [http://ploverdata.org], [the spaCy
natural language processing system](https://spacy.io/), and Python [scikit-learn](https://scikit-learn.org/stable/) machine learning package for the SVM.

`spaCy` is used to tokenize the text, then the words are filtered on the following criteria

* character length > 3
* all alphabetic
* not part of a named entity: `spaCy` doesn't always get these correctly
* not in the `spaCy` English stop list

The remaining word list is transformed into an tf/idf vector by the [sklearn](https://scikit-learn.org/stable/modules/svm.html) function `TfidfVectorizer` and then a multiclass SVM is estimated using `LinearSVC` with the default values (these are included as comments in the documentation: I have made no efforts to optimize these hyperparameters as the defaults seem to be working adequately).

The training cases were developed incrementally using an older corpus using a combination of initially seeding the cases into codeable/not codeable based on whether they had generated events, then, using the program `FJTYFilt_mark_discards.py`, manually classifying about 1000 cases into the various uncodeable categories (again, developed through a couple iterations), and finally "bootstrapping" additional training cases based on classifying unknown cases and then manually reviewing these (which is gets to be quite quick since most of the classifications are correct). 

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
8. [open: current sample uses fisheries] COVID-19
9. no codeable content: typically links and other non-language records; the corpus from which the initial training cases were derived had quite a bit of pure junk in it.

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
So, obviously, quite a few additional cases are needed in the `opinion` and `crime` categories, and `business/finance` and `accidents` could use some work as well. These additional cases can be generated from labelled cases using the `FJTYFilt_make_wordlists.py` program.


Prerequisites
=============

1. `spaCy` and the relevant routines from `sklearn` need to be installed.

2. and both use stories (not sentences) in the PLOVER data-exchange (PDE) format: as usual, the easiest way to grok this is just look
at the sample cases in *demo-REUT-20-02-25-wordlists.jsonl*

Files
=====

All programs are Python 3.7 and open source under the MIT License. The programs have been run in both Mac OS-X 10.13 and on the AWS cloud
ca. and 2018 should run without difficulty in any Unix environment. 

utilFJML.py
------------
Utility routines for the FJOLTYNG-ML system: the only routine used from this is `utilFJML.read_file`, which is a generic
routine for reading *.jsonl* files.


FJTYFilt_wordlists_from_stories.py
----------------------------------
Reads a stories file in PDE format, filters to get rid of stop words and other likely non-words, then writes a
list of the remaining words as a space-delimited string (per requirements of the sklearn SVM routines) to a PDE filename with "-wordlists" 
replacing "-stories". Input file list and file path is currently hard coded


FJTYFilt_estimator.py
---------------------
Estimate and save models using the `sklearn` modules: input and output formats are hard coded.

* TEST_RESULT_FILE_NAME = "SVM_test_results.txt"  (saves a copy of the train/test results)
* VECTORZ_PFILE_NAME = "save.vectorizer-Mk2.p"  (pickled vectorizer)
* MODEL_PFILE_NAME = "save.lin_clf-Mk2.p" (pickled SVM)

The program first does N_EXPERIMENTS (currently set at 5) train/test experiments at a 1:2 ratio (that is, model is estimated on one-third of the cases and tested on the remaining two-thirds): these results are shown on the screen and saved in the file *TEST_RESULT_FILE_NAME*. The model which is saved is estimated using all of the cases.

A classification matrix is displayed, followed by these percentages:

* category as a percent of all cases
* accuracy in classifying the category (main diagonal entry/total)
* accuracy in classifying the category as codeable or not (1 - (category-0/total))


FJTYFilt_evaluate.py
--------------------
This program classifies new cases: it reads pickled files for a vectorized and model that were generated by `FJTYFilt_estimator.py` then classifies case-word vectors from the file `INPUT_FILE_NAME` which was generated by `FJTYFilt_make_wordlists.py`. If the prediction corresponds to MODE (command-line option *m*), it writes the urls of the case to screen and a file `OUTPUT_PREFIX + "." + str(MODE) + ".urls.txt"`. A later program is then used to merge these based on the `id` field.

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
Set of training cases: most of these are mode-specific; the remaining two are a mixed set produced using `NAME.py`


FJTY_SVM_Models.zip
--------------------
Estimated SVM models: `FJTYFilt_evaluate.py` is currently set to use these


demo-REUT-20-02-25-wordlists.jsonl
----------------------------------
Sample input for `FJTYFilt_evaluate.py`



*prodigy* and other classification utilities
============================================

An article on machine learning in [*The Economist*](https://www.economist.com/technology-quarterly/2020/01/02/chinas-success-at-ai-has-relied-on-good-data) in late 2019 made the interesting observation that China's success in this area rests
not on new algorithms&mdash;Chinese machine learn enterprises use the same open source tools everyone else uses&mdash;but on their ability to quickly and inexpensively 
generate very large numbers of labelled training cases: an entire industry has arisen in China to do this.

This is also the insight behind the [explosion.ai](https://explosion.ai/) program [`prodigy`](https://prodi.gy/): enable a user or small team to rapidly 
label/classify training cases. And without the inconveniences of market authoritarianism.  `prodigy` is proprietary software but explosion.ai has certainly made far more than their share of contributions 
to open source&mdash;`spaCy` for godsakes!&mdash;and this is a place where the investment might be well worthwhile. The key contribution of
`prodigy` is the integration of a machine-learning algorithm which, well, "learns" the correct classification of your cases, so pretty
soon you are simply approving its decisions rather than having to think: this allows classification to go *very* fast.

The `FJTYFilt-plovigy.py` program below is an alternative way of doing this&mdash;it has a very small footprint, is keyboard based, and works well on 
airplanes---but does not have the machine learning component. If you are going to be doing a lot of labelling, notably in the development
of new categories, I'd recommend `prodigy`. The two programs  `name` and `name` convert between the PLOVER and `prodigy` formats.

FJTYFilt-plovigy.py
-------------------

This program is a subset of [`plovigy-mark.py`](https://github.com/openeventdata/plovigy-mark) which uses the PDE jsonl format and is a low-footprint terminal-based system for classifying discard modes. The program adds a "mode" field of the form `mode_number - mode_text`
(for example "0-codeable", "1-sports",  "2-culture/entertainment") and overwrites the 'parser', 'coder', 'codedDate' and 'codedTime' fields.

### TO RUN PROGRAM:

`python3 FJTYFilt-plovigy.py filename coder`

where the optional `filename` is the file to read with a hard-coded default; `coder` is optional coder initials.

### KEYS

* 0-9       add mode to the record and write     
* *+/space   skip: typically used when duplicates are recognized
* q         quit 

### AUTOCODING 

An autocoding file consists of a set of lines in the format

  `<mode#>-<mode_text>: <comma delimited list of phrases>`
  
#### Example:
```
0-codeable-auto: Trump, Xi, WHO
8-covid-19: coronavirus, covid-19, COVID-19
3-gold-prices: Gold prices
````

Autocoding checks the first `AUTO_WINDOW` characters in the text and if a phrase is found, the `mode` is set to `<mode#>-<mode_text>`` 
and the record is written without pausing. The lists are checked in order, so for example a text "Xi said China had the coronavirus
under control" would have a mode of `0-codeable-auto`, not `8-covid-19`.

PROGRAMMING NOTES:

1. The file FILEREC_NAME keeps track of the location in the file.

2. Output file names replaces "-stories" with "-labelled" and adds a time-stamp

3. Key input is not case-sensitive

4. With the current settings, the program uses a window 148W x 48H, measured in *characters* (not pixels)

4. Currently the program is using integers for the mode, which obviously restricts these to 10 in number. It is easy to 
   modify this to use other one-key alternatives, e.g. A, B, C,..., or if you are using a keypad, +, -, *, ... and then
   change the `FJTYFilt_wordlists.py` program to adjust for this.
   
5. AUTO_WINDOW is currently a constant but it would be easy to modify the program so this could be set in the autofile lists,
   e.g. something like
            ```
            0-codeable-auto: 128: Trump, Xi, WHO
            8-covid-19: 256: coronavirus, covid-19, COVID-19
            3-gold-prices: 32: Gold prices
            ```


FJTYFilt_wordlists_from_labelled_cases_.py
------------------------------------------
Produce word vectors based on the `FJTYFilt-plovigy.py` output. Program has three command-line options [CHECK THIS]

* <classed-files-name> <story-file-name> : names of the  mark_discards.py output and the corresponding .stories.txt file
* <output-file-prefix>: Output file is named <output-file-prefix> + "wordlists.txt" (otherwise default; this must be the final option)

There is a hard-coded list include in the program which can be used to include only categories in that list; the default is to include all categories '0' through '9'.



Additional notes
================

1. The current programs are about half-way between research and operational: file names and directories are, for the most part, 
hard-coded in the program, but it would be relatively straightforward to replace this with command-line options (see, for 
example, those implemented in `FJTYFilt_evaluate.py`) so they could be used in a scripted pipeline

2. MAKE A NOTE on configuring a pipeline to avoid running spaCy twice.

