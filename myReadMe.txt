This is bidaf code modified to output all spans (with length constraints) and their scores instead of only one span

run the code like:

nohup basic/run_single.sh /home/snigdha/data/squad/test_all.json mySensibleQAData/single_test_all_500Output.json &

**************************

PRINTING SCORES AND SPANS FOR TOP N CANDIDATES

to change the number of answer candidates you want, you could change line number 25 in basic/ensemble.py

Also, choose the appropriate function in lines 45 and 46 in basic/ensemble.py

*************************

PRINTING ALL CANDIDATES, SPANS AND THEIR SCORES

if you want to print scores and spans for all candidates of length less than a threshold, see lines 45 and 46 in basic/ensemble.py.

To set the value for threshold, chnage line 24 of basic/ensemble.py. 

Also, if you want to get rid of the thresholding on span length, see line 108 of squad/utils.py
