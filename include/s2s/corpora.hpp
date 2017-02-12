#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>

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

        unsigned int target_start_id;
        unsigned int target_end_id;
        unsigned int target_unk_id;

        void set(const s2s_options &opts){
            // resize vectors
            d_src.resize(opts.enc_feature_vocab_size.size());
            source_start_id.resize(opts.enc_feature_vocab_size.size());
            source_end_id.resize(opts.enc_feature_vocab_size.size());
            source_unk_id.resize(opts.enc_feature_vocab_size.size());
            // set start and end of sentence id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_start_id[feature_id] = d_src[feature_id].convert(opts.start_symbol);
                source_end_id[feature_id] = d_src[feature_id].convert(opts.end_symbol);
            }
            // constuct source dictionary
            std::cerr << "Reading source language training text from " << opts.srcfile << "...\n";
            freq_cut_src(opts.srcfile, d_src, opts.unk_symbol, opts.enc_feature_vocab_size);
            // set unknown id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_unk_id[feature_id] = d_src[feature_id].convert(opts.unk_symbol);
            }

            // set start and end of sentence id
            target_start_id = d_trg.convert(opts.start_symbol);
            target_end_id = d_trg.convert(opts.end_symbol);
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
        batch(){}
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
    };

    class monoling_corpus {

        public:
    
        std::vector<std::vector<std::vector<unsigned int> > > src;
        unsigned int index;
        std::vector<unsigned int> sents_order;

        monoling_corpus(){
            index = 0;
        }

        void load_src(const std::string srcfile, dicts &d){
            load_corpus_src(srcfile, d.source_start_id, d.source_end_id, d.d_src, src);
            sents_order.resize(src.size());
            std::iota(sents_order.begin(), sents_order.end(), 0);
        }

        bool next_batch_mono(batch& batch_local, const unsigned int batch_size, dicts &d){
            batch_local.set(sents_order, index, batch_size, src, d);
            if(index < src.size()){
                index += batch_local.src.at(0).at(0).size();
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

        void shuffle(){
            srand(unsigned(time(NULL)));
            std::random_shuffle(sents_order.begin(),sents_order.end());
        }

        bool next_batch_para(batch& batch_local, const unsigned int batch_size, dicts &d){
            batch_local.set(sents_order, index, batch_size, src, trg, align, d);
            if(index < src.size()){
                index += batch_local.trg.at(0).size();
                return true;
            }
            return false;
        }
    };
};

#endif
