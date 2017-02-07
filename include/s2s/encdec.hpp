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
    dynet::Parameter p_dec_init_bias;
    dynet::Parameter p_dec_init_w;
    dynet::Parameter p_Wa;
    dynet::Parameter p_Ua;
    dynet::Parameter p_va;
    dynet::Parameter p_out_R;
    dynet::Parameter p_out_bias;
    dynet::LSTMBuilder dec_builder;
    dynet::LSTMBuilder rev_enc_builder;
    dynet::LSTMBuilder fwd_enc_builder;
    dynet::expr::Expression i_Uahj;
    dynet::expr::Expression i_h_enc;
    unsigned int slen;
    const s2s_options* opts;

    explicit encoder_decoder(dynet::Model& model, const s2s_options* _opts) {
        p_feature_enc.resize(opts->enc_feature_vec_size.size());
        unsigned int num_layers = opts->num_layers;
        unsigned int rnn_size = opts->rnn_size;
        unsigned int enc_input_size = 0;
        for(unsigned int i = 0; i < opts->enc_feature_vec_size.size(); i++){
            p_feature_enc[i] = model.add_lookup_parameters(opts->enc_feature_vocab_size.at(i), {opts->enc_feature_vec_size.at(i)});
            enc_input_size += opts->enc_feature_vec_size.at(i);
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

    // build graph and return Expression for total loss
    //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
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
            h_fwd[i] = fwd_enc_builder.h.back().back();

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
            h_bwd[i] = rev_enc_builder.h.back().back();
        }
        // bidirectional encoding
        for (unsigned i = 0; i < slen; ++i) {
            h_bi[i] = concatenate(std::vector<Expression>({h_fwd[i], h_bwd[i]}));
        }
        i_h_enc = concatenate_cols(h_bi);
        Expression i_Ua = parameter(cg, p_Ua);
        i_Uahj = i_Ua * i_h_enc;
        dec_builder.new_graph(cg);
        dynet::expr::Expression i_dec_init_w = parameter(cg, p_dec_init_w);
        dec_builder.start_new_sequence(tanh(i_dec_init_w * rev_enc_builder.final_s()) + p_dec_init_bias);
    }

    dynet::expr::Expression decoder_attention(dynet::ComputationGraph& cg, const std::vector<unsigned int> prev, dynet::expr::Expression i_feed){

        dynet::expr::Expression i_x_t = lookup(cg, p_word_dec, prev);
        dynet::expr::Expression i_va = parameter(cg, p_va);
        dynet::expr::Expression i_Wa = parameter(cg, p_Wa);
        
        dynet::expr::Expression input = concatenate(std::vector<dynet::expr::Expression>({i_x_t, i_feed}));
        dynet::dec_builder.add_input(input);
        dynet::expr::Expression i_h = concatenate(dec_builder.final_h());
        dynet::expr::Expression i_wah = i_Wa * i_h;
        dynet::expr::Expression i_Wah = concatenate_cols(std::vector<dynet::expr::Expression>(slen, i_wah));
        dynet::expr::Expression i_att_pred_t = transpose(tanh(i_Wah + i_Uahj)) * i_va;

        return i_att_pred_t;

    }

    std::vector<dynet::expr::Expression> decoder_output(dynet::ComputationGraph& cg, dynet::expr::Expression i_att_pred_t){

        dynet::expr::Expression i_out_R = parameter(cg,p_out_R);
        dynet::expr::Expression i_out_bias = parameter(cg,p_out_bias);
        
        dynet::expr::Expression i_alpha_t = softmax(i_att_pred_t);
        dynet::expr::Expression i_c_t = i_h_enc * i_alpha_t;
        dynet::expr::Expression i_feed_next = concatenate(std::vector<Expression>({dec_builder.h.back().back(), i_c_t})); 
        dynet::expr::Expression i_out_pred_t = i_out_bias + i_out_R * i_feed_next;
        
        return std::vector<dynet::expr::Expression>({i_out_pred_t, i_feed_next});

    }
  
};

}

#endif
