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

#ifndef INCLUDE_GUARD_Vinyals2014_HPP
#define INCLUDE_GUARD_Vinyals2014_HPP

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
    dynet::Parameter p_Wch;
    dynet::Parameter p_out_R;
    dynet::Parameter p_out_bias;
    dynet::VanillaLSTMBuilder dec_builder;
    dynet::VanillaLSTMBuilder rev_enc_builder;
    dynet::VanillaLSTMBuilder fwd_enc_builder;

    bool rev_enc;
    bool bi_enc;
    bool dec_feed_hidden;
    bool additional_output_layer;
    bool additional_connect_layer;
    bool flag_drop_out;

    float dropout_rate_lstm_con;
    float dropout_rate_enc_in;
    float dropout_rate_enc_out;
    float dropout_rate_dec_in;
    float dropout_rate_dec_out;

    dynet::expr::Expression dropout_mask_enc_in;
    dynet::expr::Expression dropout_mask_dec_in;

    unsigned int slen;

    explicit encoder_decoder(dynet::Model& model, const s2s_options* opts) {
    
        rev_enc = opts->rev_enc;
        bi_enc = opts->bi_enc;
        dec_feed_hidden = opts->dec_feed_hidden;
        additional_output_layer = opts->additional_output_layer;
        additional_connect_layer = opts->additional_connect_layer;
        dropout_rate_lstm_con = opts->dropout_rate_lstm_con;
        dropout_rate_enc_in = opts->dropout_rate_enc_in;
        dropout_rate_dec_in = opts->dropout_rate_dec_in;

        unsigned int num_layers = opts->num_layers;
        unsigned int rnn_size = opts->rnn_size;
        unsigned int enc_input_size = 0;
        unsigned int dec_feeding_size = 0;

        flag_drop_out = true;

        assert(opts->enc_feature_vocab_size.size() == opts->enc_feature_vec_size.size());
        for(unsigned int i = 0; i < opts->enc_feature_vec_size.size(); i++){
            p_feature_enc.push_back(model.add_lookup_parameters(opts->enc_feature_vocab_size.at(i), {opts->enc_feature_vec_size.at(i)}));
            enc_input_size += opts->enc_feature_vec_size.at(i);
        }
        
        unsigned int cell_ratio = 2; // In GRU, cell_ration = 1;

        p_dec_init_w.resize(num_layers * cell_ratio);
        for(unsigned int i = 0; i < num_layers * cell_ratio; i++){
            p_dec_init_w[i] = model.add_parameters({opts->rnn_size, opts->rnn_size});
        }

        p_dec_init_bias.resize(num_layers * cell_ratio);
        for(unsigned int i = 0; i < num_layers * cell_ratio; i++){
            p_dec_init_bias[i] = model.add_parameters({opts->rnn_size});
        }

        p_word_dec = model.add_lookup_parameters(opts->dec_word_vocab_size, {opts->dec_word_vec_size}); 
        if(dec_feed_hidden){
            if(bi_enc){
                dec_feeding_size = opts->rnn_size * 3;
            }else{
                dec_feeding_size = opts->rnn_size * 2;
            }
        }else{
            if(bi_enc){
                dec_feeding_size = opts->rnn_size * 2;
            }else{
                dec_feeding_size = opts->rnn_size * 1;
            }
        }
        p_Wch = model.add_parameters({dec_feeding_size, dec_feeding_size});
        p_out_R = model.add_parameters({opts->dec_word_vocab_size, dec_feeding_size});
        p_out_bias = model.add_parameters({opts->dec_word_vocab_size});
        p_Wa = model.add_parameters({opts->att_size, opts->rnn_size});
        if(bi_enc){
            p_Ua = model.add_parameters({opts->att_size, unsigned(opts->rnn_size * 2)});
        }else{
            p_Ua = model.add_parameters({opts->att_size, unsigned(opts->rnn_size * 1)});
        }
        p_va = model.add_parameters({opts->att_size});
        rev_enc_builder = dynet::VanillaLSTMBuilder(
            num_layers,
            enc_input_size,
            rnn_size,
            model
        );
        fwd_enc_builder = dynet::VanillaLSTMBuilder(
            num_layers,
            enc_input_size,
            rnn_size,
            model
        );
        dec_builder = dynet::VanillaLSTMBuilder(
            num_layers,
            (dec_feeding_size + opts->dec_word_vec_size),
            rnn_size,
            model
        );
    }

    std::vector<dynet::expr::Expression> encoder(const batch &one_batch, dynet::ComputationGraph& cg) {

        slen = one_batch.src.size();
        const unsigned int batch_size = one_batch.src.at(0).at(0).size();

        std::vector<dynet::expr::Expression> h_fwd(slen);
        std::vector<dynet::expr::Expression> h_bwd(slen);
        std::vector<dynet::expr::Expression> h_bi(slen);

        set_dropout_mask_enc_in(cg, batch_size);
        set_dropout_mask_dec_in(cg, batch_size);
        // forward encoder
        if(rev_enc == false || bi_enc == true){
            fwd_enc_builder.new_graph(cg);
            fwd_enc_builder.start_new_sequence();
            if(flag_drop_out == true && dropout_rate_lstm_con > 0.f){
                fwd_enc_builder.set_dropout_masks(batch_size);
                // reset mask on 1-layer
                fwd_enc_builder.masks[0][0] = input(cg, {fwd_enc_builder.input_dim}, std::vector<float>(fwd_enc_builder.input_dim, 1.f));
            }
            for (unsigned int t_i = 0; t_i < slen; ++t_i) {
                assert(one_batch.src.at(t_i).size() == p_feature_enc.size());
                std::vector<dynet::expr::Expression> vec_phi(p_feature_enc.size());
                for(unsigned int f_i = 0; f_i < p_feature_enc.size(); f_i++){
                    for(unsigned int b_i = 0; b_i < one_batch.src.at(t_i).at(f_i).size(); b_i++){
                        if(!(0 <= one_batch.src.at(t_i).at(f_i).at(b_i) && one_batch.src.at(t_i).at(f_i).at(b_i) < p_feature_enc.at(f_i).dim().d[1])){
                            std::cerr << "0 < " << one_batch.src.at(t_i).at(f_i).at(b_i) << " < " << p_feature_enc.at(f_i).dim().d[1] << std::endl;
                            assert(false);
                        }
                    }
                    vec_phi[f_i] = lookup(cg, p_feature_enc[f_i], one_batch.src.at(t_i).at(f_i));
                }
                assert(one_batch.src.at(t_i).size() == vec_phi.size());
                dynet::expr::Expression i_x_t = concatenate(vec_phi);
                // dropout
                i_x_t = dynet::expr::cmult(i_x_t, dropout_mask_enc_in);
                fwd_enc_builder.add_input(i_x_t);
                h_fwd[t_i] = fwd_enc_builder.back();
            }
        }
        // backward encoder
        if(rev_enc == true || bi_enc == true){
            rev_enc_builder.new_graph(cg);
            rev_enc_builder.start_new_sequence();
            if(flag_drop_out == true && dropout_rate_lstm_con > 0.f){
                rev_enc_builder.set_dropout_masks(batch_size);
                rev_enc_builder.masks[0][0] = input(cg, {rev_enc_builder.input_dim}, std::vector<float>(rev_enc_builder.input_dim, 1.f));
            }
            for (unsigned int ind = 0; ind < slen; ++ind) {
                unsigned int t_i = (slen - 1) - ind;
                assert(t_i >= 0);
                assert(one_batch.src.at(t_i).size() == p_feature_enc.size());
                std::vector<dynet::expr::Expression> vec_phi(p_feature_enc.size());
                for(unsigned int f_i = 0; f_i < p_feature_enc.size(); f_i++){
                    for(unsigned int b_i = 0; b_i < one_batch.src.at(t_i).at(f_i).size(); b_i++){
                        if(!(0 <= one_batch.src.at(t_i).at(f_i).at(b_i) && one_batch.src.at(t_i).at(f_i).at(b_i) < p_feature_enc.at(f_i).dim().d[1])){
                            std::cerr << "0 < " << one_batch.src.at(t_i).at(f_i).at(b_i) << " < " << p_feature_enc.at(f_i).dim().d[1] << std::endl;
                            assert(false);
                        }
                    }
                    vec_phi[f_i] = lookup(cg, p_feature_enc[f_i], one_batch.src.at(t_i).at(f_i));
                }
                assert(one_batch.src.at(t_i).size() == vec_phi.size());
                dynet::expr::Expression i_x_t = concatenate(vec_phi);
                // dropout
                i_x_t = dynet::expr::cmult(i_x_t, dropout_mask_enc_in);
                rev_enc_builder.add_input(i_x_t);
                h_bwd[t_i] = rev_enc_builder.back();
            }
        }
        dynet::expr::Expression i_h_enc;
        if(bi_enc){
            // bidirectional encoding
            for (unsigned i = 0; i < slen; ++i) {
                h_bi[i] = concatenate(std::vector<Expression>({h_fwd[i], h_bwd[i]}));
            }
            i_h_enc = concatenate_cols(h_bi);
        }else{
            if(rev_enc){
                // backward encoding
                i_h_enc = concatenate_cols(h_bwd);
            }else{
                // forward encoding
                i_h_enc = concatenate_cols(h_fwd);
            }
        }
        dynet::expr::Expression i_Ua = parameter(cg, p_Ua);
        dynet::expr::Expression i_Uahj = i_Ua * i_h_enc;
        // Initialize decoder
        std::vector<dynet::expr::Expression> vec_enc_final_state;
        if(rev_enc){
            vec_enc_final_state = rev_enc_builder.final_s();
        }else{
            vec_enc_final_state = fwd_enc_builder.final_s();
        }
        dec_builder.new_graph(cg);
        if(additional_connect_layer){
            std::vector<dynet::expr::Expression> vec_dec_init_state;
            for (unsigned int i = 0; i < vec_enc_final_state.size(); i++){
                dynet::expr::Expression i_dec_init_w = parameter(cg, p_dec_init_w[i]);
                dynet::expr::Expression i_dec_init_bias = parameter(cg, p_dec_init_bias[i]);
                vec_dec_init_state.push_back(tanh(i_dec_init_w * vec_enc_final_state[i]) + i_dec_init_bias);
            }
            dec_builder.start_new_sequence(vec_dec_init_state);
        }else{
            dec_builder.start_new_sequence(vec_enc_final_state);
        }
        if(flag_drop_out == true){
            dec_builder.set_dropout_masks(batch_size);
            // reset mask on 1-layer
            dec_builder.masks[0][0] = input(cg, {dec_builder.input_dim}, std::vector<float>(dec_builder.input_dim, 1.f));
        }
        return std::vector<dynet::expr::Expression>({i_Uahj, i_h_enc});
    }
                
    std::vector<dynet::expr::Expression> init_feed(const batch &one_batch, dynet::ComputationGraph& cg){
        // std::vector<dynet::expr::Expression> i_feed{dynet::expr::zeroes(cg, dynet::Dim({dec_feeding_size}, one_batch.trg.at(0).size()))};
        std::vector<dynet::expr::Expression> i_feed{dynet::expr::zeroes(cg, dynet::Dim({p_out_R.dim().d[1]}, one_batch.trg.at(0).size()))};
        return i_feed;
    }

    dynet::expr::Expression decoder_attention(dynet::ComputationGraph& cg, const std::vector<unsigned int> prev, const dynet::expr::Expression i_feed, const dynet::expr::Expression i_Uahj){
        return decoder_attention(cg, prev, i_feed, i_Uahj, dec_builder.state());
    }

    dynet::expr::Expression decoder_attention(dynet::ComputationGraph& cg, const std::vector<unsigned int> prev, const dynet::expr::Expression i_feed, const dynet::expr::Expression i_Uahj, const dynet::RNNPointer pointer_prev){

        dynet::expr::Expression i_x_t = lookup(cg, p_word_dec, prev);
        dynet::expr::Expression i_va = parameter(cg, p_va);
        dynet::expr::Expression i_Wa = parameter(cg, p_Wa);
        
        dynet::expr::Expression i_dec_input = concatenate(std::vector<dynet::expr::Expression>({i_x_t, i_feed}));
        // dropout
        i_dec_input = dynet::expr::cmult(i_dec_input, dropout_mask_dec_in);
        dec_builder.add_input(pointer_prev, i_dec_input);
        dynet::expr::Expression i_h_dec = dec_builder.h.back().back();
        dynet::expr::Expression i_wah = i_Wa * i_h_dec;
        dynet::expr::Expression i_Wah = concatenate_cols(std::vector<dynet::expr::Expression>(slen, i_wah));
        dynet::expr::Expression i_att_pred_t = transpose(tanh(i_Wah + i_Uahj)) * i_va;

        return i_att_pred_t;

    }

    std::vector<dynet::expr::Expression> decoder_output(dynet::ComputationGraph& cg, const dynet::expr::Expression i_att_pred_t, const dynet::expr::Expression i_h_enc){

        dynet::expr::Expression i_out_R = parameter(cg, p_out_R);
        dynet::expr::Expression i_out_bias = parameter(cg, p_out_bias);
        
        dynet::expr::Expression i_alpha_t = softmax(i_att_pred_t);
        dynet::expr::Expression i_c_t = i_h_enc * i_alpha_t;
        dynet::expr::Expression i_h_dec = dec_builder.h.back().back();
        dynet::expr::Expression i_feed_next;
        if(dec_feed_hidden){
            if(additional_output_layer){
                dynet::expr::Expression i_Wch = parameter(cg, p_Wch);
                i_feed_next = tanh(i_Wch * concatenate(std::vector<Expression>({i_h_dec, i_c_t}))); 
            }else{
                i_feed_next = concatenate(std::vector<Expression>({i_h_dec, i_c_t})); 
            }
        }else{
            if(additional_output_layer){
                dynet::expr::Expression i_Wch = parameter(cg, p_Wch);
                i_feed_next = tanh(i_Wch * i_c_t); 
            }else{
                i_feed_next = i_c_t;
            }
        }
        dynet::expr::Expression i_out_pred_t = i_out_bias + i_out_R * i_feed_next;
        return std::vector<dynet::expr::Expression>({i_out_pred_t, i_feed_next});

    }

    void disable_dropout(){
        if(rev_enc == false || bi_enc == true){
            fwd_enc_builder.disable_dropout();
        }
        if(rev_enc == true || bi_enc == true){
            rev_enc_builder.disable_dropout();
        }
        dec_builder.disable_dropout();
        flag_drop_out = false;
    }

    void enable_dropout(){
        if(rev_enc == false || bi_enc == true){
            fwd_enc_builder.set_dropout(dropout_rate_lstm_con, 0.f);
        }
        if(rev_enc == true || bi_enc == true){
            rev_enc_builder.set_dropout(dropout_rate_lstm_con, 0.f);
        }
        dec_builder.set_dropout(dropout_rate_lstm_con, 0.f);
        flag_drop_out = true;
    }

    void set_dropout_mask_enc_in(dynet::ComputationGraph& cg, const unsigned int batch_size){
        if(flag_drop_out == true && dropout_rate_enc_in > 0.f){
            std::vector<dynet::expr::Expression> dropout_masks(p_feature_enc.size());
            float retention_rate = 1.f - dropout_rate_enc_in;
            float scale = 1.f / retention_rate;
            for(unsigned int f_i = 0; f_i < p_feature_enc.size(); f_i++){
                dropout_masks[f_i] = dynet::expr::random_bernoulli(cg, dynet::Dim({p_feature_enc[f_i].dim().d[0]}, batch_size), retention_rate, scale);
            }
            dropout_mask_enc_in = concatenate(dropout_masks);
        }else{
            unsigned int input_dim = 0;
            if(bi_enc ==  true || rev_enc == true){
                input_dim = rev_enc_builder.input_dim;
            }
            if(bi_enc == true || rev_enc == false){
                input_dim = fwd_enc_builder.input_dim;
            }
            dropout_mask_enc_in = input(cg, {input_dim}, std::vector<float>(input_dim, 1.f));
        }
    }

    void set_dropout_mask_dec_in(dynet::ComputationGraph& cg, const unsigned int batch_size){
        unsigned int dim_w = p_word_dec.dim().d[0];
        if(flag_drop_out == true && dropout_rate_dec_in > 0.f){
            float retention_rate = 1.f - dropout_rate_dec_in;
            float scale = 1.f / retention_rate;
            dynet::expr::Expression i_w = dynet::expr::random_bernoulli(cg, dynet::Dim({dim_w}, batch_size), retention_rate, scale);
            if(dec_feed_hidden){
                if(additional_output_layer){
                    if(bi_enc == true){
                        dynet::expr::Expression i_dec_enc_fwd_bwd = dynet::expr::random_bernoulli(cg, dynet::Dim({dec_builder.hid + fwd_enc_builder.hid + rev_enc_builder.hid}, batch_size), retention_rate, scale);
                        dropout_mask_dec_in = concatenate(std::vector<dynet::expr::Expression>({i_w, i_dec_enc_fwd_bwd}));
                    }else{
                        dynet::expr::Expression i_dec_enc_fwd = dynet::expr::random_bernoulli(cg, dynet::Dim({dec_builder.hid + fwd_enc_builder.hid}, batch_size), retention_rate, scale);
                        dropout_mask_dec_in = concatenate(std::vector<dynet::expr::Expression>({i_w, i_dec_enc_fwd}));
                    }
                }else{
                    if(bi_enc == true){
                        dynet::expr::Expression i_dec = dynet::expr::random_bernoulli(cg, dynet::Dim({dec_builder.hid}, batch_size), retention_rate, scale);
                        dynet::expr::Expression i_enc_fwd = dynet::expr::random_bernoulli(cg, dynet::Dim({fwd_enc_builder.hid}, batch_size), retention_rate, scale);
                        dynet::expr::Expression i_enc_bwd = dynet::expr::random_bernoulli(cg, dynet::Dim({rev_enc_builder.hid}, batch_size), retention_rate, scale);
                        dropout_mask_dec_in = concatenate(std::vector<dynet::expr::Expression>({i_w, i_dec, i_enc_fwd, i_enc_bwd}));
                    }else{
                        dynet::expr::Expression i_dec = dynet::expr::random_bernoulli(cg, dynet::Dim({dec_builder.hid}, batch_size), retention_rate, scale);
                        dynet::expr::Expression i_enc_fwd = dynet::expr::random_bernoulli(cg, dynet::Dim({fwd_enc_builder.hid}, batch_size), retention_rate, scale);
                        dropout_mask_dec_in = concatenate(std::vector<dynet::expr::Expression>({i_w, i_dec, i_enc_fwd}));
                    }
                }
            }else{
                if(bi_enc == true){
                    dynet::expr::Expression i_enc_fwd = dynet::expr::random_bernoulli(cg, dynet::Dim({fwd_enc_builder.hid}, batch_size), retention_rate, scale);
                    dynet::expr::Expression i_enc_bwd = dynet::expr::random_bernoulli(cg, dynet::Dim({rev_enc_builder.hid}, batch_size), retention_rate, scale);
                    dropout_mask_dec_in = concatenate(std::vector<dynet::expr::Expression>({i_w, i_enc_fwd, i_enc_bwd}));
                }else{
                    dynet::expr::Expression i_enc = dynet::expr::random_bernoulli(cg, dynet::Dim({fwd_enc_builder.hid}, batch_size), retention_rate, scale);
                    dropout_mask_dec_in = concatenate(std::vector<dynet::expr::Expression>({i_w, i_enc}));
                }
            }
        }else{
            if(dec_feed_hidden){
                if(bi_enc == true){
                    dropout_mask_dec_in = input(cg, {dim_w + dec_builder.hid + fwd_enc_builder.hid + rev_enc_builder.hid}, std::vector<float>(dim_w + dec_builder.hid + fwd_enc_builder.hid + rev_enc_builder.hid, 1.f));
                }else{
                    dropout_mask_dec_in = input(cg, {dim_w + dec_builder.hid + fwd_enc_builder.hid}, std::vector<float>(dim_w + dec_builder.hid + fwd_enc_builder.hid, 1.f));
                }
            }else{
                if(bi_enc == true){
                    dropout_mask_dec_in = input(cg, {dim_w + fwd_enc_builder.hid + rev_enc_builder.hid}, std::vector<float>(dim_w + fwd_enc_builder.hid + rev_enc_builder.hid, 1.f));
                }else{
                    dropout_mask_dec_in = input(cg, {dim_w + fwd_enc_builder.hid}, std::vector<float>(dim_w + fwd_enc_builder.hid, 1.f));
                }
            }
        }
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
        ar & rev_enc;
        ar & bi_enc;
        ar & dec_feed_hidden;
        ar & additional_output_layer;
        ar & additional_connect_layer;
        ar & dropout_rate_lstm_con;
        ar & dropout_rate_enc_in;
        ar & dropout_rate_enc_out;
        ar & dropout_rate_dec_in;
        ar & dropout_rate_dec_out;
    }
 
};

}

#endif
