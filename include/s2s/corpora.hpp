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
            cerr << "Reading source language training text from " << opts.srcfile << "...\n";
            freq_cut_src(opts.srcfile, d_src, opts.unk_symbol, opts.enc_feature_vocab_size);
            // set unknown id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_unk_id[feature_id] = d_src[feature_id].convert(opts.unk_symbol);
            }

            // set start and end of sentence id
            target_start_id = d_trg.convert(opts.start_symbol);
            target_end_id = d_trg.convert(opts.end_symbol);
            // constuct target dictionary
            cerr << "Reading target language training text from " << opts.srcfile << "...\n";
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
                cerr << "Loading source dictionary from " << file_name << "...\n";
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
            cerr << "Loading target dictionary from " << file_name << "...\n";
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
        std::vector<std::vector<unsigned int> > trg;
        std::vector<std::vector<unsigned int> > align;
        std::vector<std::vector<std::vector<unsigned int> > > src_val;
        std::vector<std::vector<unsigned int> > trg_val;
        std::vector<std::vector<unsigned int> > align_val;
        unsigned int index_train;
        unsigned int index_dev;
        std::vector<unsigned int> sents_order;
        std::vector<unsigned int> sents_dev_order;
        monoling_corpus(){
            index_train = 0;
            index_dev = 0;
        }
        void load(dicts &d, const s2s_options &opts){
            load_corpus_src(opts.srcfile, d.source_start_id, d.source_end_id, d.d_src, src);
            load_corpus_src(opts.srcvalfile, d.source_start_id, d.source_end_id, d.d_src, src_val);
            load_corpus_trg(opts.trgfile, d.target_start_id, d.target_end_id, d.d_trg, trg);
            load_corpus_trg(opts.trgvalfile, d.target_start_id, d.target_end_id, d.d_trg, trg_val);
            if(opts.guided_alignment == true){
                load_align_corpus(opts.alignfile, align);
                load_align_corpus(opts.alignfile, align_val);
                // check
                for(unsigned int sid = 0; sid < trg.size(); sid++){
                    if(trg.at(sid).size() != align.at(sid).size()){
                        cerr << "train corpus: sentence size does not match! \n";
                        assert(false);
                    }
                    for(const unsigned int tok : trg.at(sid)){
                        if(tok < 0 || src.at(sid).size() <= tok){
                            cerr << "train corpus: wrong alignment! \n";
                            assert(false);
                        }
                    }
                }
                for(unsigned int sid = 0; sid < trg_val.size(); sid++){
                    if(trg_val.at(sid).size() != align_val.at(sid).size()){
                        cerr << "dev corpus: sentence size does not match! \n";
                        assert(false);
                    }
                    for(const unsigned int tok : trg_val.at(sid)){
                        if(tok < 0 || src_val.at(sid).size() <= tok){
                            cerr << "dev corpus: wrong alignment! \n";
                            assert(false);
                        }
                    }
                }
            }
            sents_order.resize(src.size());
            std::iota(sents_order.begin(), sents_order.end(), 0);
            sents_dev_order.resize(src_val.size());
            std::iota(sents_dev_order.begin(), sents_dev_order.end(), 0);
        }
        void shuffle(){
            srand(unsigned(time(NULL)));
            std::random_shuffle(sents_order.begin(),sents_order.end());
            index_train = 0;
        }
        bool train_batch(batch& batch_local, const unsigned int batch_size, dicts &d){
            batch_local.set(sents_order, index_train, batch_size, src, trg, align, d);
            if(index_train < src.size()){
                index_train += batch_local.trg.at(0).size();
                return true;
            }
            return false;
        }
        bool dev_batch(batch& batch_local, const unsigned int batch_size, dicts &d){
            batch_local.set(sents_dev_order, index_train, batch_size, src_val, trg_val, align_val, d);
            index_dev += batch_local.trg.at(0).size();
            if(index_dev < src_val.size()){
                return true;
            }
            return false;
        }
    };

    class parallel_corpus {

        public:
    
        std::vector<std::vector<std::vector<unsigned int> > > src;
        std::vector<std::vector<unsigned int> > trg;
        std::vector<std::vector<unsigned int> > align;
        std::vector<std::vector<std::vector<unsigned int> > > src_val;
        std::vector<std::vector<unsigned int> > trg_val;
        std::vector<std::vector<unsigned int> > align_val;
        unsigned int index_train;
        unsigned int index_dev;
        std::vector<unsigned int> sents_order;
        std::vector<unsigned int> sents_dev_order;
        parallel_corpus(){
            index_train = 0;
            index_dev = 0;
        }
        void load(dicts &d, const s2s_options &opts){
            load_corpus_src(opts.srcfile, d.source_start_id, d.source_end_id, d.d_src, src);
            load_corpus_src(opts.srcvalfile, d.source_start_id, d.source_end_id, d.d_src, src_val);
            load_corpus_trg(opts.trgfile, d.target_start_id, d.target_end_id, d.d_trg, trg);
            load_corpus_trg(opts.trgvalfile, d.target_start_id, d.target_end_id, d.d_trg, trg_val);
            if(opts.guided_alignment == true){
                load_align_corpus(opts.alignfile, align);
                load_align_corpus(opts.alignfile, align_val);
                // check
                for(unsigned int sid = 0; sid < trg.size(); sid++){
                    if(trg.at(sid).size() != align.at(sid).size()){
                        cerr << "train corpus: sentence size does not match! \n";
                        assert(false);
                    }
                    for(const unsigned int tok : trg.at(sid)){
                        if(tok < 0 || src.at(sid).size() <= tok){
                            cerr << "train corpus: wrong alignment! \n";
                            assert(false);
                        }
                    }
                }
                for(unsigned int sid = 0; sid < trg_val.size(); sid++){
                    if(trg_val.at(sid).size() != align_val.at(sid).size()){
                        cerr << "dev corpus: sentence size does not match! \n";
                        assert(false);
                    }
                    for(const unsigned int tok : trg_val.at(sid)){
                        if(tok < 0 || src_val.at(sid).size() <= tok){
                            cerr << "dev corpus: wrong alignment! \n";
                            assert(false);
                        }
                    }
                }
            }
            sents_order.resize(src.size());
            std::iota(sents_order.begin(), sents_order.end(), 0);
            sents_dev_order.resize(src_val.size());
            std::iota(sents_dev_order.begin(), sents_dev_order.end(), 0);
        }
        void shuffle(){
            srand(unsigned(time(NULL)));
            std::random_shuffle(sents_order.begin(),sents_order.end());
            index_train = 0;
        }
        bool train_batch(batch& batch_local, const unsigned int batch_size, dicts &d){
            batch_local.set(sents_order, index_train, batch_size, src, trg, align, d);
            if(index_train < src.size()){
                index_train += batch_local.trg.at(0).size();
                return true;
            }
            return false;
        }
        bool dev_batch(batch& batch_local, const unsigned int batch_size, dicts &d){
            batch_local.set(sents_dev_order, index_train, batch_size, src_val, trg_val, align_val, d);
            index_dev += batch_local.trg.at(0).size();
            if(index_dev < src_val.size()){
                return true;
            }
            return false;
        }
    };
};

#endif
