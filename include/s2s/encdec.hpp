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
#include "s2s/options.hpp"
#include "s2s/corpora.hpp"

#ifndef INCLUDE_GUARD_Luong2015_HPP
#define INCLUDE_GUARD_Luong2015_HPP

namespace s2s {

class encoder_decoder {

public:
    std::vector<dynet::LookupParameter> p_feature_enc;
    dynet::LookupParameter p_word_dec;
    std::vector<dynet::Parameter> p_dec_init_bias;
    std::vector<dynet::Parameter> p_dec_init_w;
    dynet::Parameter p_Wa;
    dynet::Parameter p_Ua;
    dynet::Parameter p_va;
    dynet::Parameter p_out_R;
    dynet::Parameter p_out_bias;
    dynet::LSTMBuilder dec_builder;
    dynet::LSTMBuilder rev_enc_builder;
    dynet::LSTMBuilder fwd_enc_builder;
    unsigned int slen;

    explicit encoder_decoder(dynet::Model& model, const s2s_options* opts) {

        unsigned int num_layers = opts->num_layers;
        unsigned int rnn_size = opts->rnn_size;
        unsigned int enc_input_size = 0;

        p_feature_enc.resize(opts->enc_feature_vec_size.size());
        for(unsigned int i = 0; i < opts->enc_feature_vec_size.size(); i++){
            p_feature_enc[i] = model.add_lookup_parameters(opts->enc_feature_vocab_size.at(i), {opts->enc_feature_vec_size.at(i)});
            enc_input_size += opts->enc_feature_vec_size.at(i);
        }
        
        unsigned int cell_ratio = 2; // In GRU, cell_ration = 1;

        p_dec_init_w.resize(num_layers * cell_ratio);
        for(unsigned int i = 0; i < num_layers * cell_ratio; i++){
            p_dec_init_w[i] = model.add_parameters({opts->rnn_size});
        }

        p_dec_init_bias.resize(num_layers * cell_ratio);
        for(unsigned int i = 0; i < num_layers * cell_ratio; i++){
            p_dec_init_bias[i] = model.add_parameters({opts->rnn_size});
        }

        p_word_dec = model.add_lookup_parameters(opts->dec_word_vocab_size, {opts->dec_word_vec_size}); 
        p_out_R = model.add_parameters({opts->dec_word_vocab_size, opts->rnn_size});
        p_out_bias = model.add_parameters({opts->dec_word_vocab_size});
        p_Wa = model.add_parameters({opts->att_size, unsigned(opts->rnn_size * opts->num_layers)});
        p_Ua = model.add_parameters({opts->att_size, unsigned(opts->rnn_size * 2)});
        p_va = model.add_parameters({opts->att_size});

        rev_enc_builder = dynet::LSTMBuilder(
            num_layers,
            enc_input_size,
            rnn_size,
            model
        );
        fwd_enc_builder = dynet::LSTMBuilder(
            num_layers,
            enc_input_size,
            rnn_size,
            model
        );
        dec_builder = dynet::LSTMBuilder(
            num_layers,
            (opts->rnn_size + opts->dec_word_vec_size),
            rnn_size,
            model
        );
    }

    std::vector<dynet::expr::Expression> encoder(const batch &one_batch, dynet::ComputationGraph& cg) {
        // forward encoder
        slen = one_batch.src.size();
        fwd_enc_builder.new_graph(cg);
        fwd_enc_builder.start_new_sequence();
        std::vector<dynet::expr::Expression> h_fwd(slen);
        std::vector<dynet::expr::Expression> h_bwd(slen);
        std::vector<dynet::expr::Expression> h_bi(slen);
        for (unsigned int i = 0; i < slen; ++i) {
            std::vector<dynet::expr::Expression> phi;
            for(unsigned int f_i = 0; f_i < p_feature_enc.size(); f_i++){
                dynet::expr::Expression i_f_t = lookup(cg, p_feature_enc[f_i], one_batch.src[i][f_i]);
                phi.push_back(i_f_t);
            }
            dynet::expr::Expression i_x_t = concatenate(phi);
            fwd_enc_builder.add_input(i_x_t);
            h_fwd[i] = fwd_enc_builder.back();

        }
        // backward encoder
        rev_enc_builder.new_graph(cg);
        rev_enc_builder.start_new_sequence();
        for (unsigned int i = slen - 1; i >= 0; --i) {
            std::vector<dynet::expr::Expression> phi;
            for(unsigned int f_i = 0; f_i < p_feature_enc.size(); f_i++){
                dynet::expr::Expression i_f_t = lookup(cg, p_feature_enc[f_i], one_batch.src[i][f_i]);
                phi.push_back(i_f_t);
            }
            dynet::expr::Expression i_x_t = concatenate(phi);
            rev_enc_builder.add_input(i_x_t);
            h_bwd[i] = rev_enc_builder.back();
        }
        // bidirectional encoding
        for (unsigned i = 0; i < slen; ++i) {
            h_bi[i] = concatenate(std::vector<Expression>({h_fwd[i], h_bwd[i]}));
        }
        dynet::expr::Expression i_h_enc = concatenate_cols(h_bi);
        dynet::expr::Expression i_Ua = parameter(cg, p_Ua);
        dynet::expr::Expression i_Uahj = i_Ua * i_h_enc;
        // Initialize decoder
        dec_builder.new_graph(cg);
        std::vector<dynet::expr::Expression> vec_dec_init_state;
        std::vector<dynet::expr::Expression> vec_enc_final_state = rev_enc_builder.final_s();
        for (unsigned int i = 0; i < vec_enc_final_state.size(); i++){
            dynet::expr::Expression i_dec_init_w = parameter(cg, p_dec_init_w[i]);
            dynet::expr::Expression i_dec_init_bias = parameter(cg, p_dec_init_bias[i]);
            vec_dec_init_state.push_back(tanh(i_dec_init_w * vec_enc_final_state[i]) + i_dec_init_bias);
        }
        dec_builder.start_new_sequence(vec_dec_init_state);
        return std::vector<dynet::expr::Expression>({i_Uahj, i_h_enc});
    }

    dynet::expr::Expression decoder_attention(dynet::ComputationGraph& cg, const std::vector<unsigned int> prev, const dynet::expr::Expression i_feed, const dynet::expr::Expression i_Uahj){

        dynet::expr::Expression i_x_t = lookup(cg, p_word_dec, prev);
        dynet::expr::Expression i_va = parameter(cg, p_va);
        dynet::expr::Expression i_Wa = parameter(cg, p_Wa);
        
        dynet::expr::Expression input = concatenate(std::vector<dynet::expr::Expression>({i_x_t, i_feed}));
        dec_builder.add_input(input);
        dynet::expr::Expression i_h = concatenate(dec_builder.final_h());
        dynet::expr::Expression i_wah = i_Wa * i_h;
        dynet::expr::Expression i_Wah = concatenate_cols(std::vector<dynet::expr::Expression>(slen, i_wah));
        dynet::expr::Expression i_att_pred_t = transpose(tanh(i_Wah + i_Uahj)) * i_va;

        return i_att_pred_t;

    }

    std::vector<dynet::expr::Expression> decoder_output(dynet::ComputationGraph& cg, const dynet::expr::Expression i_att_pred_t, const dynet::expr::Expression i_h_enc){

        dynet::expr::Expression i_out_R = parameter(cg,p_out_R);
        dynet::expr::Expression i_out_bias = parameter(cg,p_out_bias);
        
        dynet::expr::Expression i_alpha_t = softmax(i_att_pred_t);
        dynet::expr::Expression i_c_t = i_h_enc * i_alpha_t;
        dynet::expr::Expression i_feed_next = concatenate(std::vector<Expression>({dec_builder.h.back().back(), i_c_t})); 
        dynet::expr::Expression i_out_pred_t = i_out_bias + i_out_R * i_feed_next;
        
        return std::vector<dynet::expr::Expression>({i_out_pred_t, i_feed_next});

    }

    void disable_dropout(){
        fwd_enc_builder.disable_dropout();
        rev_enc_builder.disable_dropout();
        dec_builder.disable_dropout();
    }

    void set_dropout(float d){
        fwd_enc_builder.set_dropout(d);
        rev_enc_builder.set_dropout(d);
        dec_builder.set_dropout(d);
    }

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & p_feature_enc;
        ar & p_word_dec;
        ar & p_dec_init_bias;
        ar & p_dec_init_w;
        ar & p_Wa;
        ar & p_Ua;
        ar & p_va;
        ar & p_out_R;
        ar & p_out_bias;
        ar & dec_builder;
        ar & rev_enc_builder;
        ar & fwd_enc_builder;
    }
 
};

}

#endif
