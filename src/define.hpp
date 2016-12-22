#include <vector>

#ifndef INCLUDE_GUARD_DEFINE_HPP
#define INCLUDE_GUARD_DEFINE_HPP

using namespace std;

using Sent = vector<int>;
using SentList = vector<Sent>;
using ParaSent = pair<Sent, Sent >;
using ParaCorp = vector<ParaSent >;
using BatchCol = vector<unsigned int>;
using Batch = vector<BatchCol>;

const int __LSTM__ = 0;
const int __FAST_LSTM__ = 1;
const int __GRU__ = 2;
const int __RNN__ = 3;

const int __Cho2014__ = 0;
const int __Sutskever2014__ = 1;
const int __Bahdanau2014__ = 2;

const int __SGD__ = 0;
const int __MomentumSGD__ = 1;
const int __Adagrad__ = 2;
const int __Adadelta__ = 3;
const int __RMSprop__ = 4;
const int __Adam__ = 5;

//parameters
int SOS_SRC;
int EOS_SRC;
int UNK_SRC;
int SOS_TRG;
int EOS_TRG;
int UNK_TRG;

#endif // INCLUDE_GUARD_DEFINE_HPP
