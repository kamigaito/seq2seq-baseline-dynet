#!/bin/sh
./bin/seq2seq \
--mode train \
--rootdir ./sample/model \
--srcfile ./sample/src.ja \
--trgfile ./sample/trg.en \
--srcvalfile ./sample/srcval.ja \
--trgvalfile ./sample/trgval.en \
--optim adadelta \
--max_batch_l 41 \
--max_length 354 \
--epochs 20 \
--enc_feature_vocab_size 4321 \
--dec_word_vocab_size 1234 \
--dec_word_vec_size 103 \
--num_layers 1 \
--rnn_size 119 \
--att_size 113 \
--dynet-mem 11000
