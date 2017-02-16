#!/bin/sh
./bin/seq2seq \
--mode predict \
--rootdir ./sample/model \
--modelfile ./sample/model/save_10.model \
--srcfile ./sample/srcval.ja \
--trgfile ./sample/model/predict.txt \
--dynet-mem 11000
