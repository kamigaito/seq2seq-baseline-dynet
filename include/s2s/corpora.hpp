#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <random>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/program_options.hpp>

#include "s2s/corpora_utils.hpp"

#ifndef INCLUDE_GUARD_S2S_DICTS_HPP
#define INCLUDE_GUARD_S2S_DICTS_HPP

namespace s2s {

    class dicts {

        public:

        std::vector<dynet::Dict> d_src;
        dynet::Dict d_trg;

        std::vector<unsigned int> source_start_id;
        std::vector<unsigned int> source_end_id;
        std::vector<unsigned int> source_unk_id;
        std::vector<unsigned int> source_pad_id;

        std::vector<unsigned int> word_freq;

        unsigned int target_start_id;
        unsigned int target_end_id;
        unsigned int target_unk_id;
        unsigned int target_pad_id;

        void set(const s2s_options &opts){
            // resize vectors
            d_src.resize(opts.enc_feature_vocab_size.size());
            source_start_id.resize(opts.enc_feature_vocab_size.size());
            source_end_id.resize(opts.enc_feature_vocab_size.size());
            source_unk_id.resize(opts.enc_feature_vocab_size.size());
            source_pad_id.resize(opts.enc_feature_vocab_size.size());
            // set start and end of sentence id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_start_id[feature_id] = d_src[feature_id].convert(opts.start_symbol);
                source_end_id[feature_id] = d_src[feature_id].convert(opts.end_symbol);
                source_pad_id[feature_id] = d_src[feature_id].convert(opts.pad_symbol);
            }
            // constuct source dictionary
            std::cerr << "Reading source language training text from " << opts.srcfile << "...\n";
            freq_cut_src(opts.srcfile, d_src, word_freq, opts.unk_symbol, opts.enc_feature_vocab_size);
            // set unknown id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_unk_id[feature_id] = d_src[feature_id].convert(opts.unk_symbol);
            }

            // set start and end of sentence id
            target_start_id = d_trg.convert(opts.start_symbol);
            target_end_id = d_trg.convert(opts.end_symbol);
            target_pad_id = d_trg.convert(opts.pad_symbol);
            // constuct target dictionary
            std::cerr << "Reading target language training text from " << opts.trgfile << "...\n";
            freq_cut_trg(opts.trgfile, d_trg, opts.unk_symbol, opts.dec_word_vocab_size);
            // set unknown id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_unk_id[feature_id] = d_src[feature_id].convert(opts.unk_symbol);
            }
        }

        void load(const s2s_options &opts){
            // resize vectors
            d_src.resize(opts.enc_feature_vocab_size.size());
            source_start_id.resize(opts.enc_feature_vocab_size.size());
            source_end_id.resize(opts.enc_feature_vocab_size.size());
            source_unk_id.resize(opts.enc_feature_vocab_size.size());
            source_pad_id.resize(opts.enc_feature_vocab_size.size());
            for(unsigned int i=0; i < d_src.size(); i++){
                std::string file_name = opts.rootdir + "/" + opts.dict_prefix + "src_" + to_string(i) + ".txt";
                std::cerr << "Loading source dictionary from " << file_name << "...\n";
                ifstream in(file_name);
                boost::archive::text_iarchive ia(in);
                ia >> d_src[i];
                in.close();
            }
            // set start and end of sentence id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_start_id[feature_id] = d_src[feature_id].convert(opts.start_symbol);
                source_end_id[feature_id] = d_src[feature_id].convert(opts.end_symbol);
                source_pad_id[feature_id] = d_src[feature_id].convert(opts.pad_symbol);
            }
            std::string file_name = opts.rootdir + "/" + opts.dict_prefix + "trg.txt";
            std::cerr << "Loading target dictionary from " << file_name << "...\n";
            ifstream in(file_name);
            boost::archive::text_iarchive ia(in);
            ia >> d_trg;
            in.close();
            // set start and end of sentence id
            target_start_id = d_trg.convert(opts.start_symbol);
            target_end_id = d_trg.convert(opts.end_symbol);
            target_pad_id = d_trg.convert(opts.pad_symbol);
        }

        void save(const s2s_options &opts){
            for(unsigned int i=0; i < d_src.size(); i++){
                std::string file_name = opts.rootdir + "/" + opts.dict_prefix + "src_" + to_string(i) + ".txt";
                std::cerr << "Writing source dictionary to " << file_name << "...\n";
                ofstream out(file_name);
                boost::archive::text_oarchive oa(out);
                oa << d_src.at(i);
                out.close();
            }
            std::string file_name = opts.rootdir + "/" + opts.dict_prefix + "trg.txt";
            cerr << "Writing target dictionary to " << file_name << "...\n";
            ofstream out(file_name);
            boost::archive::text_oarchive oa(out);
            oa << d_trg;
            out.close();
        }

    };

    class batch {
        public:
        std::vector<unsigned int> sent_id;
        std::vector<std::vector<std::vector<unsigned int> > > src;
        std::vector<std::vector<unsigned int> > trg;
        std::vector<std::vector<unsigned int> > align;
        std::vector<unsigned int> len_src;
        std::vector<unsigned int> len_trg;
        std::random_device rd;
        std::mt19937 mt;
        batch() : rd(), mt(rd()) {}
        void set(
            const std::vector<unsigned int> sents_order,
            const unsigned int index,
            const unsigned int batch_size,
            const std::vector<std::vector<std::vector<unsigned int> > > &src_input,
            const std::vector<std::vector<unsigned int> > &trg_input,
            const std::vector<std::vector<unsigned int> > &align_input,
            const dicts& d
        ){
            src = src2batch(sents_order, index, batch_size, src_input, d.source_end_id);
            trg = trg2batch(sents_order, index, batch_size, trg_input, d.target_end_id);
            align = align2batch(sents_order, index, batch_size, align_input);
            len_src = src2len(sents_order, index, batch_size, src_input);
            len_trg = trg2len(sents_order, index, batch_size, trg_input);
            sent_id.clear();
            for(unsigned int sid = 0; sid < batch_size; sid++){
                sent_id.push_back(sents_order.at(sid + index));
            }
        }
        void set(
            const std::vector<unsigned int> sents_order,
            const unsigned int index,
            const unsigned int batch_size,
            const std::vector<std::vector<std::vector<unsigned int> > > &src_input,
            const dicts& d
        ){
            src = src2batch(sents_order, index, batch_size, src_input, d.source_end_id);
            len_src = src2len(sents_order, index, batch_size, src_input);
            sent_id.clear();
            for(unsigned int sid = 0; sid < batch_size; sid++){
                sent_id.push_back(sents_order.at(sid + index));
            }
        }
        // Kiperwasser and Goldberg 2016
        void drop_word(const dicts& d, const s2s_options &opts){
            if(opts.drop_word_alpha > 0.0){
                std::uniform_real_distribution<float> prob(0.0, 1.0);
                for(unsigned int s_index = 0; s_index < src.size(); s_index++){
                    for(unsigned int b_id = 0; b_id < src.at(s_index).at(0).size(); b_id++){
                        unsigned int w_id = src.at(s_index).at(0).at(b_id);
                        if(
                            w_id != d.source_start_id.at(0) 
                            && w_id != d.source_end_id.at(0) 
                            && w_id != d.source_unk_id.at(0) 
                            && w_id != d.source_pad_id.at(0)
                        ){
                            float alpha = opts.drop_word_alpha;
                            float drop_prob = alpha / (alpha + (float)(d.word_freq[w_id]));
                            if(prob(mt) < drop_prob){
                                src[s_index][0][b_id] = d.source_unk_id.at(0);
                            }
                        }
                    }
                }
            }
        }
    };

    class monoling_corpus {

        public:
    
        std::vector<std::vector<std::vector<unsigned int> > > src;
        unsigned int index;
        std::vector<unsigned int> sents_order;
        std::vector<std::pair<unsigned int, unsigned int> > batch_order;

        monoling_corpus(){
            index = 0;
        }

        void load_src(const std::string srcfile, dicts &d){
            load_corpus_src(srcfile, d.source_start_id, d.source_end_id, d.d_src, src);
            sents_order.resize(src.size());
            std::iota(sents_order.begin(), sents_order.end(), 0);
        }

        bool next_batch_mono(batch& batch_local, dicts &d){
            if(index < batch_order.size()){
                batch_local.set(sents_order, batch_order.at(index).first, batch_order.at(index).second, src, d);
                index++;
                return true;
            }
            return false;
        }

        void reset_index(){
            index = 0;
        }

        void sort_mono_sent(const std::string shuffle_type){
            if(shuffle_type == "sort_default"){
                std::vector<std::pair<unsigned int, std::pair<unsigned int, unsigned int > > > vec_len(src.size());
                for(unsigned int sid = 0; sid < src.size(); sid++){
                    vec_len[sid].first = sid;
                    vec_len[sid].second.first = src.at(sid).size();
                }
                CompareLength comp_len;
                sort(vec_len.begin(), vec_len.end(), comp_len);
                for(unsigned int sid = 0; sid < src.size(); sid++){
                    sents_order[sid] = vec_len.at(sid).first;
                }
            }else if(shuffle_type == "default"){
            }else{
                std::cerr << "shuffle_type does not match." << std::endl;
                assert(false);
            }
        }

        void set_mono_batch_order(const unsigned int max_batch_size, const unsigned int src_tok_lim, const std::string batch_type){
            batch_order.clear();
            if(batch_type == "default"){
                unsigned int batch_start = 0;
                unsigned int batch_size = 0;
                unsigned int src_tok = 0;
                for(unsigned int sid = 0; sid < sents_order.size(); sid++){
                    unsigned int cur_len_src = src.at(sents_order.at(sid)).size();
                    if(batch_size + 1 <= max_batch_size && src_tok + cur_len_src <= src_tok_lim){
                        src_tok += cur_len_src;
                        batch_size++;
                    }else{
                        batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                        batch_start = sid;
                        src_tok = cur_len_src;
                        batch_size = 1;
                    }
                    if(sid == sents_order.size() - 1){
                        batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                    }
                }
            }else if(batch_type == "same_length"){
                unsigned int batch_start = 0;
                unsigned int batch_size = 0;
                unsigned int src_tok = 0;
                unsigned int cur_len = src.at(sents_order.at(0)).size();
                for(unsigned int sid = 0; sid < sents_order.size(); sid++){
                    unsigned int cur_len_src = src.at(sents_order.at(sid)).size();
                    if(cur_len_src == cur_len && batch_size + 1 <= max_batch_size && src_tok + cur_len_src <= src_tok_lim){
                        src_tok += cur_len_src;
                        batch_size++;
                    }else{
                        batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                        batch_start = sid;
                        cur_len = cur_len_src;
                        src_tok = cur_len_src;
                        batch_size = 1;
                    }
                    if(sid == sents_order.size() - 1){
                        batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                    }
                }
            }else{
                std::cerr << "batch_type does not match." << std::endl;
                assert(false);
            }
        }

    };

    class parallel_corpus : public monoling_corpus {

        public:
    
        std::vector<std::vector<unsigned int> > trg;
        std::vector<std::vector<unsigned int> > align;

        parallel_corpus() : monoling_corpus() {}

        void load_trg(const std::string trgfile, dicts &d){
            load_corpus_trg(trgfile, d.target_start_id, d.target_end_id, d.d_trg, trg);
        }

        void load_align(const std::string alignfile){
            load_align_corpus(alignfile, align);
        }

        void load_check(){
            // check
            assert(src.size() == trg.size()); // sentence size does not match!
        }

        void load_check_with_align(){
            assert(src.size() == trg.size() && trg.size() == align.size()); // sentence size does not match!
            // check
            for(unsigned int sid = 0; sid < trg.size(); sid++){
                assert(trg.at(sid).size() == align.at(sid).size()); // alignment size
                for(const unsigned int tok : align.at(sid)){
                    assert(0 <= tok && tok < src.at(sid).size()); // alignment range
                }
            }
        }

        void sort_para_sent(const std::string shuffle_type, const unsigned int max_batch_size, const unsigned int src_tok_lim, const unsigned int trg_tok_lim){
            if(shuffle_type == "random"){
                srand(unsigned(time(NULL)));
                std::random_shuffle(sents_order.begin(), sents_order.end());
            }else if(shuffle_type == "sort_default"){
                std::vector<std::pair<unsigned int, std::pair<unsigned int, unsigned int > > > vec_len(src.size());
                for(unsigned int sid = 0; sid < src.size(); sid++){
                    vec_len[sid].first = sid;
                    vec_len[sid].second.first = src.at(sid).size();
                    vec_len[sid].second.second = trg.at(sid).size();
                }
                CompareLength comp_len;
                sort(vec_len.begin(), vec_len.end(), comp_len);
                for(unsigned int sid = 0; sid < src.size(); sid++){
                    sents_order[sid] = vec_len.at(sid).first;
                }
            }else if(shuffle_type == "sort_random"){
                std::vector<unsigned int> sents_order_local(src.size());
                std::vector<std::pair<unsigned int, std::pair<unsigned int, unsigned int > > > vec_len(src.size());
                for(unsigned int sid = 0; sid < src.size(); sid++){
                    vec_len[sid].first = sid;
                    vec_len[sid].second.first = src.at(sid).size();
                    vec_len[sid].second.second = trg.at(sid).size();
                }
                CompareLength comp_len;
                sort(vec_len.begin(), vec_len.end(), comp_len);
                for(unsigned int sid = 0; sid < src.size(); sid++){
                    sents_order_local[sid] = vec_len.at(sid).first;
                }
                sents_order.clear();
                std::vector<unsigned int> vec_sents;
                std::pair<unsigned int, unsigned int> cur_len(src.at(sents_order_local.at(0)).size(), trg.at(sents_order_local.at(0)).size());
                for(unsigned int sid = 0; sid < sents_order_local.size(); sid++){
                    unsigned int cur_len_src = src.at(sents_order_local.at(sid)).size();
                    unsigned int cur_len_trg = trg.at(sents_order_local.at(sid)).size();
                    if(cur_len_src == cur_len.first && cur_len_trg == cur_len.second){
                        vec_sents.push_back(sents_order_local.at(sid));
                    }else{
                        cur_len.first = cur_len_src;
                        cur_len.second = cur_len_trg;
                        srand(unsigned(time(NULL)));
                        std::random_shuffle(vec_sents.begin(), vec_sents.end());
                        sents_order.insert(sents_order.end(), vec_sents.begin(), vec_sents.end());
                        vec_sents.clear();
                        vec_sents.push_back(sents_order_local.at(sid));
                    }
                    if(sid == sents_order_local.size() - 1){
                        srand(unsigned(time(NULL)));
                        std::random_shuffle(vec_sents.begin(), vec_sents.end());
                        sents_order.insert(sents_order.end(), vec_sents.begin(), vec_sents.end());
                    }
                }
            }else if(shuffle_type == "default"){
            }else{
                std::cerr << "shuffle_type does not match." << std::endl;
                assert(false);
            }
        }

        void shuffle_batch(const std::string shuffle_type){
            if(shuffle_type == "random"){
                srand(unsigned(time(NULL)));
                std::random_shuffle(batch_order.begin(), batch_order.end());
            }else if(shuffle_type == "default"){
            }else{
                std::cerr << "shuffle_type does not match." << std::endl;
                assert(false);
            }
        }

        bool next_batch_para(batch& batch_local, dicts &d){
            if(index < batch_order.size()){
                batch_local.set(sents_order, batch_order.at(index).first, batch_order.at(index).second, src, trg, align, d);
                index++;
                return true;
            }
            return false;
        }

        void set_para_batch_order(const unsigned int max_batch_size, const unsigned int src_tok_lim, const unsigned int trg_tok_lim, const std::string batch_type){
            batch_order.clear();
            if(batch_type == "default"){
                unsigned int batch_start = 0;
                unsigned int batch_size = 0;
                unsigned int src_tok = 0;
                unsigned int trg_tok = 0;
                for(unsigned int sid = 0; sid < sents_order.size(); sid++){
                    unsigned int cur_len_src = src.at(sents_order.at(sid)).size();
                    unsigned int cur_len_trg = trg.at(sents_order.at(sid)).size();
                    if(batch_size + 1 <= max_batch_size && src_tok + cur_len_src <= src_tok_lim && trg_tok + cur_len_trg <= trg_tok_lim){
                        src_tok += cur_len_src;
                        trg_tok += cur_len_trg;
                        batch_size++;
                    }else{
                        batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                        batch_start = sid;
                        src_tok = cur_len_src;
                        trg_tok = cur_len_trg;
                        batch_size = 1;
                    }
                    if(sid == sents_order.size() - 1){
                        batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                    }
                }
            }else if(batch_type == "same_length"){
                unsigned int batch_start = 0;
                unsigned int batch_size = 0;
                unsigned int src_tok = 0;
                unsigned int trg_tok = 0;
                std::pair<unsigned int, unsigned int> cur_len(src.at(sents_order.at(0)).size(), trg.at(sents_order.at(0)).size());
                for(unsigned int sid = 0; sid < sents_order.size(); sid++){
                    unsigned int cur_len_src = src.at(sents_order.at(sid)).size();
                    unsigned int cur_len_trg = trg.at(sents_order.at(sid)).size();
                    if(cur_len_src == cur_len.first && cur_len_trg == cur_len.second && batch_size + 1 <= max_batch_size && src_tok + cur_len_src <= src_tok_lim && trg_tok + cur_len_trg <= trg_tok_lim){
                        src_tok += cur_len_src;
                        trg_tok += cur_len_trg;
                        batch_size++;
                    }else{
                        batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                        batch_start = sid;
                        cur_len.first = cur_len_src;
                        cur_len.second = cur_len_trg;
                        src_tok = cur_len_src;
                        trg_tok = cur_len_trg;
                        batch_size = 1;
                    }
                    if(sid == sents_order.size() - 1){
                        batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                    }
                }
            }else{
                std::cerr << "batch_type does not match." << std::endl;
                assert(false);
            }
        }

    };
};

#endif
