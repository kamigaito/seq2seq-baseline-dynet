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
#include "s2s/encdec.hpp"

#ifndef INCLUDE_GUARD_Sutskever2014_HPP
#define INCLUDE_GUARD_Sutskever2014_HPP

using namespace std;
using namespace dynet;
using namespace dynet::expr;

template <class Builder>
class Sutskever2014 : public EncoderDecoder<Builder> {

public:
  LookupParameters* p_c;
  LookupParameters* p_ec;  // map input to embedding (used in fwd and rev models)
  Parameters* p_R;
  Parameters* p_bias;
  Builder dec_builder;
  Builder rev_enc_builder;
  boost::program_options::variables_map* vm;

  explicit Sutskever2014(Model& model, boost::program_options::variables_map* _vm) :
    dec_builder(
      vm->at("depth-layer").as<unsigned int>(),
      vm->at("dim-input").as<unsigned int>(),
      vm->at("dim-hidden").as<unsigned int>(),
      &model
    ),
    rev_enc_builder(
      vm->at("depth-layer").as<unsigned int>(),
      vm->at("dim-input").as<unsigned int>(),
      vm->at("dim-hidden").as<unsigned int>(),
      &model
    ),
    vm(_vm)
  {
    p_ec = model.add_lookup_parameters(vm->at("src-vocab-size").as<unsigned int>(), {vm->at("dim-input").as<unsigned int>()}); 
    p_c = model.add_lookup_parameters(vm->at("trg-vocab-size").as<unsigned int>(), {vm->at("dim-input").as<unsigned int>()}); 
    p_R = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>(), vm->at("dim-hidden").as<unsigned int>()});
    p_bias = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>()});
  }

  // build graph and return Expression for total loss
  //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
  virtual void Encoder(const Batch sents, ComputationGraph& cg) {
    // backward encoder
    rev_enc_builder.new_graph(cg);
    rev_enc_builder.start_new_sequence();
    for (int i = sents.size() - 1; i >= 0; --i) {
      Expression i_x_t = lookup(cg, p_ec, sents[i]);
      rev_enc_builder.add_input(i_x_t);
    }
    dec_builder.new_graph(cg);
    dec_builder.start_new_sequence(rev_enc_builder.final_s());
  }

  virtual std::vector<Expression> Decoder(ComputationGraph& cg, const BatchCol prev) {
    // decode
    Expression i_x_t = lookup(cg, p_c, prev);
    Expression i_y_t = dec_builder.add_input(i_x_t);
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);
    Expression i_r_t = i_bias + i_R * i_y_t;

    return std::vector<Expression>({i_r_t});
  }

};

#endif
