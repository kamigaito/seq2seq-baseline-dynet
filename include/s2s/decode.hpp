#include "dynet/nodes.h"
#include "dynet/exec.h"
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
#include "s2s/metrics.hpp"

namespace s2s {

    void greedy_decode(const batch& one_batch, std::vector<std::vector<unsigned int > >& osent, encoder_decoder *encdec, dicts &d, const s2s_options &opts){
        //unsigned slen = sents.size();
        dynet::ComputationGraph cg;
        osent.push_back(std::vector<unsigned int>(one_batch.src.at(0).at(0).size(), d.target_start_id));
        std::vector<dynet::expr::Expression> i_enc = encdec->encoder(one_batch, cg);
        std::vector<dynet::expr::Expression> i_feed = encdec->init_feed(one_batch, cg);
        for (int t = 0; t < opts.max_length; ++t) {
            dynet::expr::Expression i_att_t = encdec->decoder_attention(cg, osent[t], i_feed[t], i_enc[0]);
            std::vector<dynet::expr::Expression> i_out_t = encdec->decoder_output(cg, i_att_t, i_enc[1]);
            i_feed.push_back(i_out_t[1]);
            Expression predict = softmax(i_out_t[0]);
            std::vector<dynet::Tensor> results = cg.incremental_forward(predict).batch_elems();
            std::vector<unsigned int> osent_col;
            for(unsigned int i = 0; i < results.size(); i++){
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

    std::string print_sents(std::vector<std::vector<unsigned int > >& osent, dicts& d){
        std::string sents = "";
        std::vector<std::vector<unsigned int> > sents_conved;
        sents_conved.resize(osent.at(0).size());
        for(unsigned int col_id = 0; col_id < osent.size(); col_id++){
            for(unsigned int sid = 0; sid < osent.at(col_id).size(); sid++){
                sents_conved[sid].push_back(osent.at(col_id).at(sid));
            }
        }
        for(const auto sent : sents_conved){
            for(const auto wid : sent){
                std::string word = d.d_trg.convert(wid);
                sents += word;
                if(wid == d.target_end_id){
                    break;
                }
                sents += " ";
            }
            sents += "\n";
        }
        return sents;
    }

};
