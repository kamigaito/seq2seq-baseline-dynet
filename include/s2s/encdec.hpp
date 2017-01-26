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

#ifndef INCLUDE_GUARD_Bahdanau2014_HPP
#define INCLUDE_GUARD_Bahdanau2014_HPP

namespace s2s {

template <class Builder>
class encoder_decoder {

public:
    dynet::LookupParameters* p_word_enc;
    dynet::LookupParameters* p_word_dec;  // map input to embedding (used in fwd and rev models)
    dynet::Parameters* p_out_R;
    dynet::Parameters* p_att_R;
    dynet::Parameters* p_out_bias;
    dynet::Parameters* p_att_bias;
    dynet::Parameters* p_Wa;
    dynet::Parameters* p_Ua;
    dynet::Parameters* p_va;
    Builder* dec_builder;
    Builder* rev_enc_builder;
    Builder* fwd_enc_builder;
    dynet::Expression i_Uahj;
    dynet::Expression i_h_enc;
    unsigned int slen;
    boost::program_options::variables_map* vm;

    explicit encoder_decoder(Model& model, boost::program_options::variables_map* _vm) :
        vm(_vm)
    {
        p_c = model.add_lookup_parameters(vm->at("trg-vocab-size").as<unsigned int>(), {vm->at("dim-hidden").as<unsigned int>()}); 
        p_ec = model.add_lookup_parameters(vm->at("src-vocab-size").as<unsigned int>(), {vm->at("dim-hidden").as<unsigned int>()}); 
        p_R = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>(), vm->at("dim-hidden").as<unsigned int>()});
        p_bias = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>()});
        p_Wa = model.add_parameters({vm->at("dim-attention").as<unsigned int>(), unsigned(vm->at("dim-hidden").as<unsigned int>() * vm->at("depth-layer").as<unsigned int>())});
        p_Ua = model.add_parameters({vm->at("dim-attention").as<unsigned int>(), unsigned(vm->at("dim-hidden").as<unsigned int>() * 2 * vm->at("depth-layer").as<unsigned int>())});
        p_va = model.add_parameters({vm->at("dim-attention").as<unsigned int>()});
        Builder* dec_builder = new Builder(
            _vm->at("depth-layer").as<unsigned int>(),
            _vm->at("dim-input").as<unsigned int>(),
            _vm->at("dim-hidden").as<unsigned int>(),
            &model
        ),
        Builder* rev_enc_builder = new Builder(
            _vm->at("depth-layer").as<unsigned int>(),
            _vm->at("dim-input").as<unsigned int>(),
            _vm->at("dim-hidden").as<unsigned int>(),
            &model
        ),
        Builder* fwd_enc_builder = new Builer(
            _vm->at("depth-layer").as<unsigned int>(),
            _vm->at("dim-input").as<unsigned int>(),
            _vm->at("dim-hidden").as<unsigned int>(),
            &model
        ),
    }

    // build graph and return Expression for total loss
    //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
    std::vector<dynet::Expression> encoder(const Batch sents, dynet::ComputationGraph& cg) {
        // forward encoder
        slen = sents.size();
        fwd_enc_builder->new_graph(cg);
        fwd_enc_builder->start_new_sequence();
        std::vector<Expression> h_fwd(sents.size());
        std::vector<Expression> h_bwd(sents.size());
        std::vector<Expression> h_bi(sents.size());
        for (unsigned i = 0; i < sents.size(); ++i) {
            Expression i_x_t = lookup(cg, p_word_enc, sents[i]);
            //h_fwd[i] = fwd_enc_builder.add_input(i_x_t);
            fwd_enc_builder->add_input(i_x_t);
            h_fwd[i] = concatenate(fwd_enc_builder->final_h());
        }
        // backward encoder
        rev_enc_builder->new_graph(cg);
        rev_enc_builder->start_new_sequence();
        for (int i = sents.size() - 1; i >= 0; --i) {
            Expression i_x_t = lookup(cg, p_word_enc, sents[i]);
            //h_bwd[i] = rev_enc_builder.add_input(i_x_t);
            rev_enc_builder->add_input(i_x_t);
            h_bwd[i] = concatenate(rev_enc_builder->final_h());
        }
        // bidirectional encoding
        for (unsigned i = 0; i < sents.size(); ++i) {
            h_bi[i] = concatenate(std::vector<Expression>({h_fwd[i], h_bwd[i]}));
        }
        i_h_enc = concatenate_cols(h_bi);
        Expression i_Ua = parameter(cg, p_Ua);
        i_Uahj = i_Ua * i_h_enc;
        dec_builder.new_graph(cg);
        dec_builder.start_new_sequence(rev_enc_builder->final_s());
    }

    dynet::Expression decoder_attention(ComputationGraph& cg, const BatchCol prev, dynet::Expression i_feed){

        Expression i_x_t = lookup(cg, p_word_dec, prev);
        Expression i_va = parameter(cg, p_va);
        Expression i_Wa = parameter(cg, p_Wa);
        
        Expression input = concatenate(std::vector<dynet::Expression>({i_x_t, i_feed}));
        dec_builder.add_input(input);
        Expression i_h = concatenate(dec_builder.final_h());
        Expression i_wah = i_Wa * i_h;
        Expression i_Wah = concatenate_cols(std::vector<dynet::Expression>(slen, i_wah));
        Expression i_e_t = transpose(tanh(i_Wah + i_Uahj)) * i_va + i_att_bias;

        return i_e_t

    }

    std::vector<dynet::Expression> decoder_output(ComputationGraph& cg, dynet::Expression i_e_t){

        Expression i_out_R = parameter(cg,p_out_R);
        Expression i_out_bias = parameter(cg,p_out_bias);
        
        Expression i_alpha_t = softmax(i_e_t);
        Expression i_c_t = i_h_enc * i_alpha_t;
        Expression i_feed_next = concatenate(std::vector<Expression>({concatenate(dec_builder.final_h()), i_c_t})); 
        Expression i_out_pred_t = i_out_bias + i_out_R * i_feed_next;
        
        return std::vector<dynet::Expression>({i_out_pred_t, i_feed_next});

    }
  
};

}

#endif // INCLUDE_GUARD_HOGE_HPP
