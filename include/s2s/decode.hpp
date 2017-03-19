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
                unsigned int w_id = 0;
                float w_prob = -FLT_MAX;
                if(osent.back().at(i) == d.target_end_id || t == opts.max_length - 1){
                    w_id = d.target_end_id;
                }else{
                    for(unsigned int j=0; j<output.size(); j++){
                        if(output[j] > w_prob){
                            w_id = j;
                            w_prob = output[j];
                        }
                    }
                }
                osent_col.push_back(w_id);
            }
            osent.push_back(osent_col);
            // end check
            unsigned int num_end = 0;
            for(const unsigned int w_id : osent_col){
                if(w_id == d.target_end_id){
                    num_end++;
                }
            }
            if(num_end == osent_col.size()){
                break;
            }
        }
    }

    void greedy_decode_vinyals(const batch& one_batch, std::vector<std::vector<unsigned int > >& osent, encoder_decoder *encdec, dicts &d, const s2s_options &opts){
        //unsigned slen = sents.size();
        dynet::ComputationGraph cg;
        std::vector<unsigned int> XX_count = one_batch.len_src;
        osent.push_back(std::vector<unsigned int>(one_batch.src.at(0).at(0).size(), d.target_start_id));
        std::vector<dynet::expr::Expression> i_enc = encdec->encoder(one_batch, cg);
        std::vector<dynet::expr::Expression> i_feed = encdec->init_feed(one_batch, cg);
        // skip start and end symbols from the count
        for(unsigned int i=0; i < XX_count.size(); i++){
            XX_count[i] -= 2;
        }
        for (unsigned int t = 0; t < opts.max_length; ++t) {
            dynet::expr::Expression i_att_t = encdec->decoder_attention(cg, osent[t], i_feed[t], i_enc[0]);
            std::vector<dynet::expr::Expression> i_out_t = encdec->decoder_output(cg, i_att_t, i_enc[1]);
            i_feed.push_back(i_out_t[1]);
            Expression predict = softmax(i_out_t[0]);
            std::vector<dynet::Tensor> results = cg.incremental_forward(predict).batch_elems();
            std::vector<unsigned int> osent_col;
            for(unsigned int i = 0; i < results.size(); i++){
                auto output = as_vector(results.at(i));
                unsigned int w_id = 0;
                float w_prob = -FLT_MAX;
                if(osent.back().at(i) == d.target_end_id || t == opts.max_length - 1){
                    w_id = d.target_end_id;
                }else if(XX_count.at(i) >= ((opts.max_length - t) - 1)){
                    w_id = d.d_trg.convert("XX");
                }else{
                    for(unsigned int j=0; j < output.size(); j++){
                        if(XX_count.at(i) > 0){
                            if(j != d.target_end_id){
                                if(output[j] > w_prob){
                                    w_id = j;
                                    w_prob = output[j];
                                }
                            }
                        }else if(XX_count.at(i) == 0){
                            std::string w_str = d.d_trg.convert(j);
                            if(j != d.d_trg.convert("XX") && w_str[0] != '('){
                                if(output[j] > w_prob){
                                    w_id = j;
                                    w_prob = output[j];
                                }
                            }
                        }else{
                            std::cerr << "Count does not match." << std::endl;
                            assert(false);
                        }
                    }
                }
                osent_col.push_back(w_id);
                if(w_id == d.d_trg.convert("XX")){
                    XX_count[i]--;
                }
            }
            osent.push_back(osent_col);
            // end check
            unsigned int num_end = 0;
            for(const unsigned int w_id : osent_col){
                if(w_id == d.target_end_id){
                    num_end++;
                }
            }
            if(num_end == osent_col.size()){
                break;
            }
        }
    }


    void beam_decode(){
    }

    std::vector<std::string> print_sents(std::vector<std::vector<unsigned int > >& osent, dicts& d){
        std::vector<std::string> str_sents;
        std::vector<std::vector<unsigned int> > sents_conved;
        sents_conved.resize(osent.at(0).size());
        for(unsigned int col_id = 0; col_id < osent.size(); col_id++){
            for(unsigned int sid = 0; sid < osent.at(col_id).size(); sid++){
                sents_conved[sid].push_back(osent.at(col_id).at(sid));
            }
        }
        for(const auto sent : sents_conved){
            std::string str_sent = "";
            for(const auto wid : sent){
                std::string word = d.d_trg.convert(wid);
                str_sent += word;
                if(wid == d.target_end_id){
                    break;
                }
                str_sent += " ";
            }
            str_sents.push_back(str_sent);
        }
        return str_sents;
    }

};
