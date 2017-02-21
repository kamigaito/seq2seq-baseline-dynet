#!/bin/sh
./bin/seq2seq \
--mode predict \
--rootdir ./sample/model \
--modelfile ./sample/model/save_10.model \
--srcfile ./sample/srcval.ja \
--trgfile ./sample/model/predict.txt \
--max_batch_pred 8 \
--dynet-mem 11000
