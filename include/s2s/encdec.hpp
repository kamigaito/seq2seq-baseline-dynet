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

#ifndef INCLUDE_GUARD_Luong2015_HPP
#define INCLUDE_GUARD_Luong2015_HPP

namespace s2s {

template <class Builder>
class encoder_decoder {

public:
    std::vector<dynet::LookupParameters*> p_feature_enc;
    dynet::LookupParameters* p_word_dec;
    dynet::Parameters* p_dec_init_bias;
    dynet::Parameters* p_dec_init_w;
    dynet::Parameters* p_Wa;
    dynet::Parameters* p_Ua;
    dynet::Parameters* p_va;
    dynet::Parameters* p_out_R;
    dynet::Parameters* p_out_bias;
    Builder* dec_builder;
    Builder* rev_enc_builder;
    Builder* fwd_enc_builder;
    dynet::Expression i_Uahj;
    dynet::Expression i_h_enc;
    unsigned int slen;
    const options* opts;

    explicit encoder_decoder(Model& model, const options* _opts){
        p_feature_enc.resize(opts->enc_feature_vec_size.size());
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
        Builder* rev_enc_builder = new Builder(
            opts->num_layers,
            enc_input_size,
            opts->rnn_size,
            &model
        ),
        Builder* fwd_enc_builder = new Builer(
            opts->num_layers,
            enc_input_size,
            opts->rnn_size,
            &model
        ),
        Builder* dec_builder = new Builder(
            opts->num_layers,
            (opts->rnn_size + opts->dec_word_vec_size),
            opts->rnn_size,
            &model
        ),
    }

    // build graph and return Expression for total loss
    //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
    std::vector<dynet::Expression> encoder(const batch &one_batch, dynet::ComputationGraph& cg) {
        // forward encoder
        slen = one_batch.src.size();
        fwd_enc_builder->new_graph(cg);
        fwd_enc_builder->start_new_sequence();
        std::vector<dynet::Expression> h_fwd(slen);
        std::vector<dynet::Expression> h_bwd(slen);
        std::vector<dynet::Expression> h_bi(slen);
        for (unsigned int i = 0; i < slen; ++i) {
            std::vector<dynet::Expression> phi;
            for(unsigned int f_i = 0; f_i < p_feature_enc.size(); f_i++){
                dynet::Expression i_f_t = lookup(cg, p_feature_enc[f_i], one_batch.src[i][f_i]);
                phi.push_back(i_f_t);
            }
            dynet::Expression i_x_t = concatenate(phi);
            fwd_enc_builder->add_input(i_x_t);
            h_fwd[i] = fwd_enc_builder->h.back().back();

        }
        // backward encoder
        rev_enc_builder->new_graph(cg);
        rev_enc_builder->start_new_sequence();
        for (unsigned int i = slen - 1; i >= 0; --i) {
            std::vector<dynet::Expression> phi;
            for(unsigned int f_i = 0; f_i < p_feature_enc.size(); f_i++){
                dynet::Expression i_f_t = lookup(cg, p_feature_enc[f_i], one_batch.src[i][f_i]);
                phi.push_back(i_f_t);
            }
            dynet::Expression i_x_t = concatenate(phi);
            rev_enc_builder->add_input(i_x_t);
            h_bwd[i] = rev_enc_builder->h.back().back();
        }
        // bidirectional encoding
        for (unsigned i = 0; i < sents.size(); ++i) {
            h_bi[i] = concatenate(std::vector<Expression>({h_fwd[i], h_bwd[i]}));
        }
        i_h_enc = concatenate_cols(h_bi);
        Expression i_Ua = parameter(cg, p_Ua);
        i_Uahj = i_Ua * i_h_enc;
        dec_builder.new_graph(cg);
        dec_builder.start_new_sequence(tanh(p_dec_init_w * rev_enc_builder->final_s()) + p_dec_init_bias);
    }

    dynet::Expression decoder_attention(ComputationGraph& cg, const std::vector<unsigned int> prev, dynet::Expression i_feed){

        dynet::Expression i_x_t = lookup(cg, p_word_dec, prev);
        dynet::Expression i_va = parameter(cg, p_va);
        dynet::Expression i_Wa = parameter(cg, p_Wa);
        
        dynet::Expression input = concatenate(std::vector<dynet::Expression>({i_x_t, i_feed}));
        dynet::dec_builder.add_input(input);
        dynet::Expression i_h = concatenate(dec_builder.final_h());
        dynet::Expression i_wah = i_Wa * i_h;
        dynet::Expression i_Wah = concatenate_cols(std::vector<dynet::Expression>(slen, i_wah));
        dynet::Expression i_att_pred_t = transpose(tanh(i_Wah + i_Uahj)) * i_va;

        return i_att_pred_t;

    }

    std::vector<dynet::Expression> decoder_output(ComputationGraph& cg, dynet::Expression i_att_pred_t){

        dynet::Expression i_out_R = parameter(cg,p_out_R);
        dynet::Expression i_out_bias = parameter(cg,p_out_bias);
        
        dynet::Expression i_alpha_t = softmax(i_att_pred_t);
        dynet::Expression i_c_t = i_h_enc * i_alpha_t;
        dynet::Expression i_feed_next = concatenate(std::vector<Expression>({dec_builder.h.back().back(), i_c_t})); 
        dynet::Expression i_out_pred_t = i_out_bias + i_out_R * i_feed_next;
        
        return std::vector<dynet::Expression>({i_out_pred_t, i_feed_next});

    }
  
};

}

#endif
