#!/bin/sh
./bin/seq2seq \
--mode train \
--rootdir ./sample/model \
--srcfile ./sample/src.ja \
--trgfile ./sample/trg.en \
--srcvalfile ./sample/srcval.ja \
--trgvalfile ./sample/trgval.en \
--optim adam \
--max_batch_l 8 \
--max_length 300 \
--epochs 20 \
--enc_feature_vocab_size 5000 \
--dec_word_vocab_size 5000 \
--num_layers 1 \
--rnn_size 128 \
--att_size 128 \
--dynet-mem 11000
