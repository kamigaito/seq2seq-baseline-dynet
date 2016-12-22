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
// this is same to dynet/example/encdec.cc
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/define.hpp"
#include "s2s/encdec.hpp"

#ifndef INCLUDE_GUARD_Cho2014_HPP
#define INCLUDE_GUARD_Cho2014_HPP

using namespace std;
using namespace dynet;
using namespace dynet::expr;

template <class Builder>
class Cho2014 : public EncoderDecoder<Builder> {
public:
  LookupParameters* p_c;
  LookupParameters* p_ec;  // map input to embedding (used in fwd and rev models)
  Parameters* p_ie2h;
  Parameters* p_bie;
  Parameters* p_h2oe;
  Parameters* p_boe;
  Parameters* p_R;
  Parameters* p_bias;
  Builder dec_builder;
  Builder rev_enc_builder;
  Builder fwd_enc_builder;
  boost::program_options::variables_map* vm;

  explicit Cho2014(Model& model, boost::program_options::variables_map* _vm) :
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
    fwd_enc_builder(
      vm->at("depth-layer").as<unsigned int>(),
      vm->at("dim-input").as<unsigned int>(),
      vm->at("dim-hidden").as<unsigned int>(),
      &model
    ),
    vm(_vm)
  {
    p_ie2h = model.add_parameters({unsigned(vm->at("dim-hidden").as<unsigned int>() * vm->at("depth-layer").as<unsigned int>() * 1.5), unsigned(vm->at("dim-hidden").as<unsigned int>() * vm->at("depth-layer").as<unsigned int>() * 2)});
    p_bie = model.add_parameters({unsigned(vm->at("dim-hidden").as<unsigned int>() * vm->at("depth-layer").as<unsigned int>() * 1.5)});
    p_h2oe = model.add_parameters({unsigned(vm->at("dim-hidden").as<unsigned int>() * vm->at("depth-layer").as<unsigned int>()), unsigned(vm->at("dim-hidden").as<unsigned int>() * vm->at("depth-layer").as<unsigned int>() * 1.5)});
    p_boe = model.add_parameters({unsigned(vm->at("dim-hidden").as<unsigned int>() * vm->at("depth-layer").as<unsigned int>())});
    p_c = model.add_lookup_parameters(vm->at("trg-vocab-size").as<unsigned int>(), {vm->at("dim-input").as<unsigned int>()}); 
    p_ec = model.add_lookup_parameters(vm->at("src-vocab-size").as<unsigned int>(), {vm->at("dim-input").as<unsigned int>()}); 
    p_R = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>(), vm->at("dim-hidden").as<unsigned int>()});
    p_bias = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>()});
  }

  // build graph and return Expression for total loss
  //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
  virtual void Encoder(const Batch sents, ComputationGraph& cg) {
    // forward encoder
    fwd_enc_builder.new_graph(cg);
    fwd_enc_builder.start_new_sequence();
    for (const auto input : sents) {
      Expression i_x_t = lookup(cg, p_ec, input);
      fwd_enc_builder.add_input(i_x_t);
    }
    // backward encoder
    rev_enc_builder.new_graph(cg);
    rev_enc_builder.start_new_sequence();
    for (int i = sents.size() - 1; i >= 0; --i) {
      Expression i_x_t = lookup(cg, p_ec, sents[i]);
      rev_enc_builder.add_input(i_x_t);
    }
    
    // encoder -> decoder transformation
    vector<Expression> to;
    for (auto h_l : fwd_enc_builder.final_h()) to.push_back(h_l);
    for (auto h_l : rev_enc_builder.final_h()) to.push_back(h_l);
    
    Expression i_combined = concatenate(to);
    Expression i_ie2h = parameter(cg, p_ie2h);
    Expression i_bie = parameter(cg, p_bie);
    Expression i_t = i_bie + i_ie2h * i_combined;
    cg.incremental_forward();
    Expression i_h = rectify(i_t);
    Expression i_h2oe = parameter(cg,p_h2oe);
    Expression i_boe = parameter(cg,p_boe);
    Expression i_nc = i_boe + i_h2oe * i_h;
    
    vector<Expression> oein1, oein2, oein;
    for (unsigned i = 0; i < vm->at("depth-layer").as<unsigned int>(); ++i) {
      oein1.push_back(pickrange(i_nc, i * vm->at("dim-hidden").as<unsigned int>(), (i + 1) * vm->at("dim-hidden").as<unsigned int>()));
      oein2.push_back(tanh(oein1[i]));
    }
    for (unsigned i = 0; i < vm->at("depth-layer").as<unsigned int>(); ++i) oein.push_back(oein1[i]);
    for (unsigned i = 0; i < vm->at("depth-layer").as<unsigned int>(); ++i) oein.push_back(oein2[i]);

    dec_builder.new_graph(cg);
    dec_builder.start_new_sequence(oein);
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
