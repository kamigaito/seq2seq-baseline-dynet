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

#include "define.hpp"

#ifndef INCLUDE_GUARD_ENC_DEC_HPP
#define INCLUDE_GUARD_ENC_DEC_HPP

using namespace std;
using namespace dynet;
using namespace dynet::expr;

template <class Builder>
class EncoderDecoder {
public:
/*public:
  LookupParameters* p_c;
  LookupParameters* p_ec;  // map input to embedding (used in fwd and rev models)
  //Parameters* p_ie2h;
  //Parameters* p_bie;
  //Parameters* p_h2oe;
  //Parameters* p_boe;
  Parameters* p_R;
  Parameters* p_bias;
  Parameters* p_zero;
  Builder dec_builder;
  Builder rev_enc_builder;
  boost::program_options::variables_map* vm;
*/
  EncoderDecoder(){}
/*
  explicit EncoderDecoder(Model& model, boost::program_options::variables_map* _vm) :
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
    //vm = _vm;
    //p_ie2h = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5), unsigned(HIDDEN_DIM * LAYERS * 2)});
    //p_bie = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5)});
    //p_h2oe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS), unsigned(HIDDEN_DIM * LAYERS * 1.5)});
    //p_boe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS)});
    p_ec = model.add_lookup_parameters(vm->at("src-vocab-size").as<unsigned int>(), {vm->at("dim-input").as<unsigned int>()}); 
    p_c = model.add_lookup_parameters(vm->at("trg-vocab-size").as<unsigned int>(), {vm->at("dim-input").as<unsigned int>()}); 
    p_R = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>(), vm->at("dim-hidden").as<unsigned int>()});
    p_bias = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>()});
    p_zero = model.add_parameters({vm->at("dim-input").as<unsigned int>()});
  }
*/
  // build graph and return Expression for total loss
  //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
  virtual void Encoder(const Batch sents, ComputationGraph& cg) {
/*
    // backward encoder
    rev_enc_builder.new_graph(cg);
    rev_enc_builder.start_new_sequence();
    for (int i = sents.size() - 1; i >= 0; --i) {
      Expression i_x_t = lookup(cg, p_ec, sents[i]);
      rev_enc_builder.add_input(i_x_t);
    }
    dec_builder.new_graph(cg);
    dec_builder.start_new_sequence(rev_enc_builder.final_s());
*/
  }


  virtual Expression Decoder(ComputationGraph& cg, const BatchCol prev) {
/*
    // decode
    Expression i_x_t = lookup(cg, p_c, prev);
    Expression i_y_t = dec_builder.add_input(i_x_t);
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);
    Expression i_r_t = i_bias + i_R * i_y_t;
    return i_r_t;
*/
  }
/*
  virtual void GreedyDecode(const Batch& sents, SentList& osents, ComputationGraph &cg){
    unsigned bsize = sents.at(0).size();
    //unsigned slen = sents.size();
    Encoder(sents, cg);
    Decoder(cg);
    Batch prev(1);
    osents.resize(bsize);
    for(unsigned int bi=0; bi < bsize; bi++){
      osents[bi].push_back(SOS_TRG);
      prev[0].push_back((unsigned int)SOS_TRG);
    }
    for (int t = 0; t < vm->at("limit-length").as<unsigned int>(); ++t) {
      unsigned int end_count = 0;
      for(unsigned int bi=0; bi < bsize; bi++){
        if(osents[bi][t] == EOS_TRG){
          end_count++;
        }
      }
      if(end_count == bsize) break;
      Expression i_r_t = Decoder(cg, prev[t]);
      Expression predict = softmax(i_r_t);
      vector<Tensor> results = cg.incremental_forward().batch_elems();
      prev.resize(t+2);
      for(unsigned int bi=0; bi < bsize; bi++){
        auto output = as_vector(results.at(bi));
        int w_id = 0;
        if(osents[bi][t] == EOS_TRG){
          w_id = EOS_TRG;
        }else{
          double w_prob = output[w_id];
          for(unsigned int j=0; j<output.size(); j++){
            double j_prob = output[j];
            if(j_prob > w_prob){
              w_id = j;
              w_prob = j_prob;
            }
          }
        }
        osents[bi].push_back(w_id);
        prev[t+1].push_back((unsigned int)w_id);
      }
    }
  }
*/
};

#endif
