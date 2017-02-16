#!/bin/sh
./bin/seq2seq \
--mode train \
--rootdir ./sample/model \
--srcfile ./sample/src.ja \
--trgfile ./sample/trg.en \
--srcvalfile ./sample/srcval.ja \
--trgvalfile ./sample/trgval.en \
--optim adadelta \
--max_batch_l 1 \
--max_length 300 \
--epochs 20 \
--enc_feature_vocab_size 40000 \
--dec_word_vocab_size 61 \
--dec_word_vec_size 512 \
--num_layers 3 \
--rnn_size 256 \
--att_size 256 \
--dynet-mem 11000
