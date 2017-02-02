#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#ifndef INCLUDE_GUARD_S2S_DICTS_HPP
#define INCLUDE_GUARD_S2S_DICTS_HPP

namespace s2s {

    class dicts {

        public:

        std::vector<dynet::Dict> d_src
        dynet::Dict d_trg;

        std::vector<unsigned int> source_start_id;
        std::vector<unsigned int> source_end_id;
        std::vector<unsigned int> source_unk_id;

        unsigned int target_start_id;
        unsigned int target_end_id;
        unsigned int target_unk_id;

        void set(const options &opts){
            // resize vectors
            d_src.resize(opts.enc_feature_vocab_size.size());
            source_start_id.resize(opts.enc_feature_vocab_size.size());
            source_end_id.resize(opts.enc_feature_vocab_size.size());
            source_unk_id.resize(opts.enc_feature_vocab_size.size());
            // set start and end of sentence id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_start_id[feature_id] = d_src[feature_id].Convert(opts.start_symbol);
                source_end_id[feature_id] = d_src[feature_id].Convert(opts.end_symbol);
            }
            // constuct source dictionary
            cerr << "Reading source language training text from " << opts.srcfile << "...\n";
            freq_cut(opts.srcfile, d_src, opts.unk_symbol, opts.enc_feature_vocab_size);
            // set unknown id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_unk_id[feature_id] = d_src[feature_id].Convert(opts.unk_symbol);
            }

            // set start and end of sentence id
            target_start_id = d_trg.Convert(opts.start_symbol);
            target_end_id = d_trg.Convert(opts.end_symbol);
            // constuct target dictionary
            cerr << "Reading target language training text from " << opts.srcfile << "...\n";
            freq_cut(opts.trgfile, d_trg, opts.unk_symbol, opts.target_vocab_size);
            // set unknown id
            source_unk_id = d_src.Convert(opts.unk_symbol);

        }

        void load(const options &opts){
            // resize vectors
            d_src.resize(opts.enc_feature_vocab_size.size());
            source_start_id.resize(opts.enc_feature_vocab_size.size());
            source_end_id.resize(opts.enc_feature_vocab_size.size());
            source_unk_id.resize(opts.enc_feature_vocab_size.size());
            for(unsigned int i=0; i < d_src.size(); i++){
                str::string file_name = opts.roodir + "/" + opts.dict_prefix + "src_" + to_string(i) + ".txt";
                cerr << "Loading source dictionary from " << file_name << "...\n";
                ifstream in(file_name);
                boost::archive::text_iarchive ia(out);
                ia >> d_src.at(i);
                in.close();
            }
            // set start and end of sentence id
            for(unsigned int feature_id = 0; feature_id < d_src.size(); feature_id++){
                source_start_id[feature_id] = d_src[feature_id].Convert(opts.start_symbol);
                source_end_id[feature_id] = d_src[feature_id].Convert(opts.end_symbol);
            }
            str::string file_name = opts.roodir + "/" + opts.dict_prefix + "trg.txt";
            cerr << "Loading target dictionary from " << file_name << "...\n";
            ifstream in(file_name);
            boost::archive::text_iarchive ia(out);
            ia >> d_trg;
            in.close();
            // set start and end of sentence id
            target_start_id = d_trg.Convert(opts.start_symbol);
            target_end_id = d_trg.Convert(opts.end_symbol);
        }

        void save(const options &opts){
            for(unsigned int i=0; i < d_src.size(); i++){
                str::string file_name = opts.roodir + "/" + opts.dict_prefix + "src_" + to_string(i) + ".txt";
                cerr << "Writing source dictionary to " << file_name << "...\n";
                ofstream out(file_name);
                boost::archive::text_oarchive oa(out);
                oa << d_src.at(i);
                out.close();
            }
            str::string file_name = opts.roodir + "/" + opts.dict_prefix + "trg.txt";
            cerr << "Writing target dictionary to " << file_name << "...\n";
            ofstream out(file_name);
            boost::archive::text_oarchive oa(out);
            oa << d_trg;
            out.close();
        }

    }

    class parallel_corpus {

        public:
    
        std::vector<std::vector<std::vector<unsigned int> > > src;
        std::vector<std::vector<unsigned int> > trg;
        std::vector<std::vector<unsigned int> > align;
        std::vector<std::vector<unsigned int> > src_val;
        std::vector<std::vector<unsigned int> > trg_val;
        std::vector<std::vector<unsigned int> > align_val;
        unsigned int index_train;
        unsigned int index_dev;
        std::vector<unsigned int> sents_order;
        parallel_corpus(){
            index_train = 0;
            index_dev = 0;
        }
        load(const dicts &d, const options &opts){
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
            std::iota(sents_order.begin(),sents_order.end(),0);
        }
        void shuffle(){
            srand(unsigned(time(NULL)));
            std::random_shuffle(sents_order.begin(),sents_order.end());
        }
        bool train_status(){
            if(index_train < src.size()){
                return true;
            }
            return false;
        }
        batch train_batch(unsigned int batch_size){
            batch batch_local(index_train, batch_size, src, trg, align, d);
            return batch_local;
        }
    }

    class batch {
        public:
        std::vector<std::vector<std::vector<unsigned int> > > src;
        std::vector<std::vector<unsigned int> > trg;
        std::vector<std::vector<unsigned int> > align;
        batch(
              const unsigned int index,
              const unsigned int batch_size,
              const std::vector<std::vector<unsigned int> > &src,
              const std::vector<std::vector<unsigned int> > &trg,
              const std::vector<std::vector<unsigned int> > &align,
              const dicts& d
        ){
        
        }
    }

    std::vector<std::vector<unsigned int > > vec_row2col(const unsigned int index, const unsigned int batch_size, const std::vector<std::vector<unsigned int> > &vec_input){
    
    }
};
#endif
