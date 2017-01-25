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

namespace s2s {

    template <class Builder>
    void greedy_decode(const Sent& sent, Sent& osent, EncoderDecoder<Builder> *encdec, ComputationGraph &cg, boost::program_options::variables_map &vm){
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

    void beam_decode(){
    }

};
