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

#include "s2s/encdec.hpp"
#include "s2s/define.hpp"
#include "s2s/comp.hpp"
#include "s2s/preprocess.hpp"
#include "s2s/metrics.hpp"

namespace Decode {

template <class Builder>
void Greedy(const Sent& sent, Sent& osent, EncoderDecoder<Builder> *encdec, ComputationGraph &cg, boost::program_options::variables_map &vm){
  //unsigned slen = sents.size();
	Batch batch_sent;
	SentToBatch(sent, batch_sent);
  encdec->Encoder(batch_sent, cg);
  //encdec->Decoder(cg);
	osent.push_back(SOS_TRG);
  for (int t = 1; t < vm.at("length-limit").as<unsigned int>(); ++t) {
		BatchCol batch_col;
		batch_col.push_back(osent[t-1]);
    Expression i_r_t = encdec->Decoder(cg, batch_col);
    Expression predict = softmax(i_r_t);
    std::vector<Tensor> results = cg.incremental_forward().batch_elems();
    auto output = as_vector(results.at(0));
    int w_id = 0;
    double w_prob = output[w_id];
    for(unsigned int j=0; j<output.size(); j++){
      if(output[j] > w_prob){
        w_id = j;
        w_prob = output[j];
      }
    }
    osent.push_back(w_id);
    if(osent[t] == EOS_TRG){
			break;
    }
  }
}
/*
template <class Builder>
void Greedy(const Batch& sents, SentList& osents, EncoderDecoder<Builder> *encdec, ComputationGraph &cg, boost::program_options::variables_map &vm){
  unsigned bsize = sents.at(0).size();
  //unsigned slen = sents.size();
  encdec->Encoder(sents, cg);
  encdec->Decoder(cg);
  Batch prev(1);
  osents.resize(bsize);
  for(unsigned int bi=0; bi < bsize; bi++){
    osents[bi].push_back(SOS_TRG);
    prev[0].push_back((unsigned int)SOS_TRG);
  }
  for (int t = 0; t < vm.at("length-limit").as<unsigned int>(); ++t) {
    unsigned int end_count = 0;
    for(unsigned int bi=0; bi < bsize; bi++){
      if(osents[bi][t] == EOS_TRG){
        end_count++;
      }
    }
    if(end_count == bsize) break;
    Expression i_r_t = encdec->Decoder(cg, prev[t]);
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
void Beam(){
}

};
