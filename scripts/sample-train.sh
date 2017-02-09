#!/bin/sh
./bin/seq2seq \
--rootdir ./sample/model
--srcfile ./sample/src.ja
--trgfile ./sample/trg.en
--srcvalfile ./sample/srcval.ja
--trgvalfile ./sample/trgval.en
