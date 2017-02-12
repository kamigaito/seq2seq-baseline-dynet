#!/bin/sh
./bin/seq2seq \
--mode train \
--rootdir ./sample/model \
--srcfile ./sample/src.ja \
--trgfile ./sample/trg.en \
--srcvalfile ./sample/srcval.ja \
--trgvalfile ./sample/trgval.en \
--optim sgd \
--max_batch_l 64 \
--max_length 40 \
--epochs 20 \
--dec_word_vocab_size 61 \
--num_layers 1 \
--rnn_size 64 \
--dynet-mem 11000
