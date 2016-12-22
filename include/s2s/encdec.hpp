// Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems. 2014.
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/fast-lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/define.hpp"

#ifndef INCLUDE_GUARD_ENC_DEC_HPP
#define INCLUDE_GUARD_ENC_DEC_HPP

using namespace std;
using namespace dynet;
using namespace dynet::expr;

template <class Builder>
class EncoderDecoder {
public:
  EncoderDecoder(){}
  // build graph and return Expression for total loss
  //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
  virtual void Encoder(const Batch sents, ComputationGraph& cg) {
  }

  virtual vector<Expression> Decoder(ComputationGraph& cg, const BatchCol prev) {
  }
};

#endif
