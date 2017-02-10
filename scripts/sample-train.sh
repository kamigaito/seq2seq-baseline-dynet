#!/bin/sh
./bin/seq2seq \
--mode train \
--rootdir ./sample/model \
--srcfile ./sample/src.ja \
--trgfile ./sample/trg.en \
--srcvalfile ./sample/srcval.ja \
--trgvalfile ./sample/trgval.en \
--optim sgd \
--max_batch_l 8 \
--dynet-mem 8096
