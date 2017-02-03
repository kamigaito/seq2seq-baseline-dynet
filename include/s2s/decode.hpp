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
    void greedy_decode(const batch& one_batch, std::vector<std::vector<unsigned int > >& osent, encoder_decoder *encdec, ComputationGraph &cg, const dicts &d, const options &opts){
        //unsigned slen = sents.size();
        encdec->encoder(one_batch.src, cg);
        osent.push_back(std::vector<unsigned int>(d.target_start_id, one_batch.at(0).size()));
        dynet::Expression i_feed;
        for (int t = 1; t < opts.max_length; ++t) {
            Expression i_att_t = encdec->decoder_attention(cg, osent[t-1], i_feed);
            std::vector<dynet::Expression> i_out_t = encdec->decoder_output(cg, i_att_t);
            i_feed = i_out_t[1];
            Expression predict = softmax(i_out_t[0]);
            std::vector<dynet::Tensor> results = cg.incremental_forward().batch_elems();
            std::vector<unsigned int> osent_col;
            for(unsigned int i=0; results.size(); i++){
                auto output = as_vector(results.at(i));
                int w_id = 0;
                double w_prob = output[w_id];
                for(unsigned int j=0; j<output.size(); j++){
                    if(output[j] > w_prob){
                        w_id = j;
                        w_prob = output[j];
                    }
                }
                osent_col.push_back(w_id);
            }
            osent.push_back(osent_col);
        }
    }

    void beam_decode(){
    }

    std::string print_sents(std::vector<std::vector<unsigned int > >& osent, const dicts& d){
        std::string sents = "";
        std::vector<std::vector<unsigned int> > sent_conved;
        sents_conved.resize(osent.size());
        for(unsigned int col_id = 0; col_id < osent.size(); col_id++){
            for(unsigned int sid = 0; osent.at(col_id).size(); sid++){
                sents_conved[sid].push_back(osent.at(col_id).at(sid));
            }
        }
        for(const auto sent : sents_conved){
            for(const auto wid : sent){
                std::string word = d_trg.Convert(wid);
                sents += word;
                if(wid == d.target_end_id){
                    break;
                }
                sents += " ";
            }
            sents += std::endl;
        }
        return sents;
    }

};
