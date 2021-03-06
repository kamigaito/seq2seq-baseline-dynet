#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#ifndef INCLUDE_GUARD_S2S_OPTIONS_HPP
#define INCLUDE_GUARD_S2S_OPTIONS_HPP

namespace s2s {

    struct s2s_options {
        public:
        std::string mode;
        std::string rootdir;
        std::string srcfile;
        std::string trgfile;
        std::string srcvalfile;
        std::string trgvalfile;
        std::string alignfile;
        std::string alignvalfile;
        std::string save_file;
        std::string modelfile;
        std::string dict_prefix;
        //std::string rootdir;
        unsigned int num_layers;
        unsigned int rnn_size;
        unsigned int att_size;
        bool shared_input;
        bool dec_feed_hidden;
        bool additional_output_layer;
        bool additional_connect_layer;
        bool bi_enc;
        bool rev_enc;
        bool clipping_enabled;
        std::string enc_feature_vec_size_str;
        std::string enc_feature_vocab_size_str;
        std::vector<unsigned int> enc_feature_vec_size;
        std::vector<unsigned int> enc_feature_vocab_size;
        unsigned int dec_word_vec_size;
        unsigned int dec_word_vocab_size;
        bool guided_alignment;
        float guided_alignment_weight;
        float guided_alignment_decay;
        float guided_output_weight;
        float guided_output_decay;
        unsigned int guided_alignment_start_epoch;
        unsigned int guided_alignment_decay_for_each;
        unsigned int epochs;
        unsigned int start_epoch;
        unsigned int decay_for_each;
        unsigned int sgd_start_epoch;
        unsigned int sgd_start_decay;
        unsigned int sgd_start_decay_for_each;
        std::string optim;
        bool lr_auto_decay;
        float learning_rate;
        float lr_decay;
        float sgd_start_learning_rate;
        float sgd_start_lr_decay;
        float dropout_rate_lstm_con;
        float dropout_rate_enc_in;
        float dropout_rate_dec_in;
        float clip_threshold;
        float drop_word_alpha;
        unsigned int max_batch_train;
        unsigned int max_batch_pred;
        unsigned int max_length;
        unsigned int src_tok_lim_train;
        unsigned int trg_tok_lim_train;
        unsigned int src_tok_lim_pred;
        unsigned int trg_tok_lim_pred;
        std::string start_symbol;
        std::string end_symbol;
        std::string unk_symbol;
        std::string pad_symbol;
        std::string sort_sent_type_train;
        std::string batch_type_train;
        std::string shuffle_batch_type_train;
        std::string sort_sent_type_pred;
        std::string batch_type_pred;
        std::string shuffle_batch_type_pred;
        std::string decoder_type;
        unsigned int save_every;
        unsigned int print_every;
        unsigned int seed;
        s2s_options(){
            mode = "";
            rootdir = "";
            srcfile = "";
            trgfile = "";
            srcvalfile = "";
            trgvalfile = "";
            alignfile = "";
            alignvalfile = "";
            save_file = "save_";
            dict_prefix = "dict_";
            num_layers = 3;
            rnn_size = 256;
            att_size = 256;
            shared_input = false;
            clipping_enabled = false;
            /*
            enc_feature_vec_size;
            enc_feature_vocab_size;
            */
            enc_feature_vec_size_str = "256";
            enc_feature_vocab_size_str = "20000";
            dec_word_vec_size = 128;
            dec_word_vocab_size = 20000;
            guided_alignment = false;
            guided_alignment_weight = 0.5;
            guided_alignment_decay = 0.9;
            guided_alignment_start_epoch = 5;
            guided_alignment_decay_for_each = 5;
            guided_output_weight = 1.0;
            guided_output_decay = 1.0;
            dec_feed_hidden = false;
            additional_output_layer = false;
            additional_connect_layer = false;
            bi_enc = true;
            rev_enc = true;
            clipping_enabled = true;
            epochs = 20;
            start_epoch = 5;
            decay_for_each = 5;
            sgd_start_epoch = UINT_MAX;
            sgd_start_decay = UINT_MAX;
            sgd_start_decay_for_each = UINT_MAX;
            lr_auto_decay = false;
            learning_rate = 1.0;
            lr_decay = 1.0;
            sgd_start_learning_rate = 0.1;
            sgd_start_lr_decay = 1.0;
            optim = "sgd";
            dropout_rate_lstm_con = 0.3;
            dropout_rate_enc_in = 0.3;
            dropout_rate_dec_in = 0.3;
            clip_threshold = 5.0;
            drop_word_alpha = 0.0;
            max_batch_train = 32;
            max_batch_pred = 32;
            max_length = 300;
            src_tok_lim_train = 500;
            trg_tok_lim_train = 500;
            src_tok_lim_pred = 500;
            trg_tok_lim_pred = 500;
            sort_sent_type_train = "same_length";
            batch_type_train = "default";
            shuffle_batch_type_train = "default";
            sort_sent_type_pred = "same_length";
            batch_type_pred = "default";
            shuffle_batch_type_pred = "default";
            decoder_type = "greedy";
            start_symbol = "<s>";
            end_symbol = "</s>";
            unk_symbol = "<unk>";
            pad_symbol = "<pad>";
            save_every = 1;
            print_every = 1;
            seed = 1;
        }
private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & save_file;
            ar & dict_prefix;
            ar & num_layers;
            ar & rnn_size;
            ar & att_size;
            ar & shared_input;
            ar & dec_feed_hidden;
            ar & additional_output_layer;
            ar & additional_connect_layer;
            ar & bi_enc;
            ar & rev_enc;
            ar & enc_feature_vec_size;
            ar & enc_feature_vocab_size;
            ar & dec_word_vec_size;
            ar & dec_word_vocab_size;
            ar & guided_alignment;
            ar & guided_alignment_weight;
            ar & guided_alignment_decay;
            ar & guided_alignment_start_epoch;
            ar & guided_alignment_decay_for_each;
            ar & guided_output_weight;
            ar & guided_output_decay;
            ar & epochs;
            ar & start_epoch;
            ar & optim;
            ar & learning_rate;
            ar & dropout_rate_lstm_con;
            ar & dropout_rate_enc_in;
            ar & dropout_rate_dec_in;
            ar & lr_decay;
            ar & clip_threshold;
            ar & clipping_enabled;
            ar & drop_word_alpha;
            ar & start_symbol;
            ar & end_symbol;
            ar & unk_symbol;
            ar & pad_symbol;
            ar & save_every;
            ar & print_every;
            ar & seed;
        }
    };

    void set_s2s_options(boost::program_options::options_description *bpo, s2s_options *opts) {
        namespace po = boost::program_options;
        bpo->add_options()
        ("dynet-mem", po::value<unsigned int>()->default_value(8096), "mem")
        ("dynet-weight-decay", po::value<float>()->default_value(0.0), "l2 decay")
        ("mode", po::value<std::string>(&(opts->mode))->required(), "select from 'train', 'predict' or 'test'")
        ("rootdir", po::value<std::string>(&(opts->rootdir))->required(), "source train file")
        ("srcfile", po::value<std::string>(&(opts->srcfile))->required(), "source train file")
        ("trgfile", po::value<std::string>(&(opts->trgfile))->required(), "source train file")
        ("srcvalfile", po::value<std::string>(&(opts->srcvalfile)), "source train file")
        ("trgvalfile", po::value<std::string>(&(opts->trgvalfile)), "source train file")
        ("alignfile", po::value<std::string>(&(opts->alignfile)), "source train file")
        ("alignvalfile", po::value<std::string>(&(opts->alignvalfile)), "source train file")
        ("modelfile", po::value<std::string>(&(opts->modelfile)), "source train file")
        ("save_file_prefix", po::value<std::string>(&(opts->save_file))->default_value("save"), "source train file")
        ("dict_prefix", po::value<std::string>(&(opts->dict_prefix))->default_value("dict_"), "source train file")
        ("num_layers", po::value<unsigned int>(&(opts->num_layers))->default_value(3), "test input")
        ("rnn_size", po::value<unsigned int>(&(opts->rnn_size))->default_value(256), "test input")
        ("att_size", po::value<unsigned int>(&(opts->att_size))->default_value(256), "batch size")
        ("shared_input", po::value<bool>(&(opts->shared_input))->default_value(false), "batch size")
        ("dec_feed_hidden", po::value<bool>(&(opts->dec_feed_hidden))->default_value(false), "batch size")
        ("additional_output_layer", po::value<bool>(&(opts->additional_output_layer))->default_value(true), "batch size")
        ("additional_connect_layer", po::value<bool>(&(opts->additional_connect_layer))->default_value(true), "batch size")
        ("bi_enc", po::value<bool>(&(opts->bi_enc))->default_value(true), "batch size")
        ("rev_enc", po::value<bool>(&(opts->rev_enc))->default_value(true), "batch size")
        ("clipping_enabled", po::value<bool>(&(opts->clipping_enabled))->default_value(true), "batch size")
        ("enc_feature_vec_size", po::value<std::string>(&(opts->enc_feature_vec_size_str))->default_value("256"), "target train file")
        ("enc_feature_vocab_size", po::value<std::string>(&(opts->enc_feature_vocab_size_str))->default_value("20000"), "target train file")
        ("dec_word_vec_size", po::value<unsigned int>(&(opts->dec_word_vec_size)), "batch size")
        ("dec_word_vocab_size", po::value<unsigned int>(&(opts->dec_word_vocab_size)), "batch size")
        ("guided_alignment", po::value<bool>(&(opts->guided_alignment))->default_value(false), "batch size")
        ("guided_alignment_weight", po::value<float>(&(opts->guided_alignment_weight))->default_value(0.3), "batch size")
        ("guided_alignment_decay", po::value<float>(&(opts->guided_alignment_decay))->default_value(0.5), "batch size")
        ("guided_alignment_start_epoch", po::value<unsigned int>(&(opts->guided_alignment_start_epoch))->default_value(5), "batch size")
        ("guided_alignment_decay_for_each", po::value<unsigned int>(&(opts->guided_alignment_decay_for_each))->default_value(5), "batch size")
        ("guided_output_weight", po::value<float>(&(opts->guided_output_weight))->default_value(1.0), "batch size")
        ("guided_output_decay", po::value<float>(&(opts->guided_output_decay))->default_value(1.0), "batch size")
        ("epochs", po::value<unsigned int>(&(opts->epochs))->default_value(20), "batch size")
        ("start_epoch", po::value<unsigned int>(&(opts->start_epoch))->default_value(5), "batch size")
        ("decay_for_each", po::value<unsigned int>(&(opts->decay_for_each))->default_value(5), "batch size")
        ("sgd_start_epoch", po::value<unsigned int>(&(opts->sgd_start_epoch))->default_value(UINT_MAX), "batch size")
        ("sgd_start_decay", po::value<unsigned int>(&(opts->sgd_start_decay))->default_value(UINT_MAX), "batch size")
        ("sgd_start_decay_for_each", po::value<unsigned int>(&(opts->sgd_start_decay_for_each))->default_value(UINT_MAX), "batch size")
        ("optim", po::value<std::string>(&(opts->optim))->default_value("sgd"), "source train file")
        ("sgd_start_learning_rate", po::value<float>(&(opts->sgd_start_learning_rate))->default_value(0.1), "batch size")
        ("sgd_start_lr_decay", po::value<float>(&(opts->sgd_start_lr_decay))->default_value(1.0), "batch size")
        ("lr_auto_decay", po::value<bool>(&(opts->lr_auto_decay))->default_value(false), "batch size")
        ("learning_rate", po::value<float>(&(opts->learning_rate))->default_value(0.1), "batch size")
        ("lr_decay", po::value<float>(&(opts->lr_decay))->default_value(1.0), "batch size")
        ("dropout_rate_lstm_con", po::value<float>(&(opts->dropout_rate_lstm_con))->default_value(0.3), "batch size")
        ("dropout_rate_enc_in", po::value<float>(&(opts->dropout_rate_enc_in))->default_value(0.3), "batch size")
        ("dropout_rate_dec_in", po::value<float>(&(opts->dropout_rate_dec_in))->default_value(0.3), "batch size")
        ("clip_threshold", po::value<float>(&(opts->clip_threshold))->default_value(5.0), "batch size")
        ("drop_word_alpha", po::value<float>(&(opts->drop_word_alpha))->default_value(0.0), "batch size")
        ("max_batch_train", po::value<unsigned int>(&(opts->max_batch_train))->default_value(32), "batch size")
        ("max_batch_pred", po::value<unsigned int>(&(opts->max_batch_pred))->default_value(1), "batch size")
        ("max_length", po::value<unsigned int>(&(opts->max_length))->default_value(300), "batch size")
        ("start_symbol", po::value<std::string>(&(opts->start_symbol))->default_value("<s>"), "source train file")
        ("end_symbol", po::value<std::string>(&(opts->end_symbol))->default_value("</s>"), "source train file")
        ("unk_symbol", po::value<std::string>(&(opts->unk_symbol))->default_value("<unk>"), "source train file")
        ("pad_symbol", po::value<std::string>(&(opts->pad_symbol))->default_value("<pad>"), "source train file")
        ("src_tok_lim_train", po::value<unsigned int>(&(opts->src_tok_lim_train)), "batch size")
        ("trg_tok_lim_train", po::value<unsigned int>(&(opts->trg_tok_lim_train)), "batch size")
        ("src_tok_lim_pred", po::value<unsigned int>(&(opts->src_tok_lim_pred)), "batch size")
        ("trg_tok_lim_pred", po::value<unsigned int>(&(opts->trg_tok_lim_pred)), "batch size")
        ("sort_sent_type_train", po::value<std::string>(&(opts->sort_sent_type_train))->default_value("default"), "source train file")
        ("batch_type_train", po::value<std::string>(&(opts->batch_type_train))->default_value("sent_length"), "source train file")
        ("shuffle_batch_type_train", po::value<std::string>(&(opts->shuffle_batch_type_train))->default_value("default"), "source train file")
        ("sort_sent_type_pred", po::value<std::string>(&(opts->sort_sent_type_pred))->default_value("default"), "source train file")
        ("batch_type_pred", po::value<std::string>(&(opts->batch_type_pred))->default_value("sent_length"), "source train file")
        ("shuffle_batch_type_pred", po::value<std::string>(&(opts->shuffle_batch_type_pred))->default_value("default"), "source train file")
        ("decoder_type", po::value<std::string>(&(opts->decoder_type))->default_value("greedy"), "source train file")
        ("save_every", po::value<unsigned int>(&(opts->save_every))->default_value(1), "batch size")
        ("print_every", po::value<unsigned int>(&(opts->print_every))->default_value(1), "batch size")
        ("seed", po::value<unsigned int>(&(opts->seed))->default_value(0), "batch size");
    }

    void add_s2s_options_train(const boost::program_options::variables_map &vm, s2s_options *opts){
        std::vector<std::string> vec_str_enc_feature_vec_size;
        boost::algorithm::split_regex(vec_str_enc_feature_vec_size, opts->enc_feature_vec_size_str, boost::regex(","));
        for(auto feature_vec_size : vec_str_enc_feature_vec_size){
            opts->enc_feature_vec_size.push_back(std::stoi(feature_vec_size));
        }
        std::vector<std::string> vec_str_enc_feature_vocab_size;
        boost::algorithm::split_regex(vec_str_enc_feature_vocab_size, opts->enc_feature_vocab_size_str, boost::regex(","));
        for(auto feature_vocab_size : vec_str_enc_feature_vocab_size){
            opts->enc_feature_vocab_size.push_back(std::stoi(feature_vocab_size));
        }
        assert(opts->enc_feature_vocab_size.size() == opts->enc_feature_vec_size.size());
    }

    bool check_s2s_options_train(const boost::program_options::variables_map &vm, const s2s_options &opts){
    
    }

    bool check_s2s_options_predict(const boost::program_options::variables_map &vm, const s2s_options &opts){
    
    }
};

#endif
