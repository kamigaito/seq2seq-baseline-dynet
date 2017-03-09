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
        std::vector<std::vector<std::vector<unsigned int> > > src;
        std::vector<std::vector<unsigned int> > trg;
        std::vector<std::vector<unsigned int> > align;
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
        }
        void set(
              const std::vector<unsigned int> sents_order,
              const unsigned int index,
              const unsigned int batch_size,
              const std::vector<std::vector<std::vector<unsigned int> > > &src_input,
              const dicts& d
        ){
            src = src2batch(sents_order, index, batch_size, src_input, d.source_end_id);
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
        std::vector<unsigned int> batch_order;

        monoling_corpus(){
            index = 0;
        }

        void load_src(const std::string srcfile, const unsigned int max_batch_size, dicts &d){
            load_corpus_src(srcfile, d.source_start_id, d.source_end_id, d.d_src, src);
            sents_order.resize(src.size());
            std::iota(sents_order.begin(), sents_order.end(), 0);
            unsigned int batch_size = (src.size() + max_batch_size - 1) / max_batch_size;
            batch_order.resize(batch_size);
            for(unsigned int i = 0; i < batch_order.size(); i++){
                batch_order[i] = i * max_batch_size;
            }
        }

        bool next_batch_mono(batch& batch_local, const unsigned int max_batch_size, dicts &d){
            if(index < batch_order.size()){
                batch_local.set(sents_order, batch_order.at(index), max_batch_size, src, d);
                index++;
                return true;
            }
            return false;
        }
        void reset_index(){
            index = 0;
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

        void shuffle_sent(){
            srand(unsigned(time(NULL)));
            std::random_shuffle(sents_order.begin(), sents_order.end());
        }

        void shuffle_batch(){
            srand(unsigned(time(NULL)));
            std::random_shuffle(batch_order.begin(), batch_order.end());
        }

        bool next_batch_para(batch& batch_local, const unsigned int max_batch_size, dicts &d){
            if(index < batch_order.size()){
                batch_local.set(sents_order, batch_order.at(index), max_batch_size, src, trg, align, d);
                index++;
                return true;
            }
            return false;
        }
    };
};

#endif
