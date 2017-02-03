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
            index_train = 0;
        }
        bool train_status(){
            if(index_train < src.size()){
                return true;
            }
            return false;
        }
        bool dev_status(){
            if(index_dev < src_val.size()){
                return true;
            }
            return false;
        }
        batch train_batch(const unsigned int batch_size, const dicts &d){
            batch batch_local(index_train, batch_size, src, trg, align, d);
            index_train += batch.trg.at(0).size();
            return batch_local;
        }
        batch dev_batch(const unsigned int batch_size, const dicts &d){
            batch batch_local(index_train, batch_size, src_val, trg_val, align_val, d);
            index_dev += batch.trg.at(0).size();
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
              const std::vector<std::vector<std::vector<unsigned int> > > &src,
              const std::vector<std::vector<unsigned int> > &trg,
              const std::vector<std::vector<unsigned int> > &align,
              const dicts& d
        ){
            src = src2batch();
            trg = trg2batch();
            align = align2batch();
        }
    }

    std::vector<std::vector<unsigned int > > trg2batch(const unsigned int index, const unsigned int max_batch_size, const std::vector<std::vector<unsigned int> > &vec_input, const unsigned int eos){
        unsigned int max_len = 0;
        unsigned int batch_size = 0;
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < vec_input.size(); sid++){
            batch_size++;
            unsigned int cur_len = vec_input.at(sid + index).size();
            if(cur_len > max_len){
                max_len = cur_len;
            }
        }
        std::vector<std::vector<unsigned int>> col(max_len, std::vector<unsigned int>(batch_size, eos));
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < vec_input.size(); sid++){
            for(unsigned int pos = 0; pos < vec_input.at(sid + index).size(); pos++){
                col[pos][sid] = vec_input.at(sid + index).at(pos);
            }
        }
        return col;
    }

    std::vector<std::vector<unsigned int > > src2batch(const unsigned int index, const unsigned int batch_size, const std::vector<std::vector<std::vector<unsigned int> > > &vec_input, const std::vector<unsigned int> vec_eos){
        unsigned int max_len = 0;
        unsigned int batch_size = 0;
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < vec_input.size(); sid++){
            batch_size++;
            unsigned int cur_len = vec_input.at(sid + index).size();
            if(cur_len > max_len){
                max_len = cur_len;
            }
        }
        std::vector<std::vector<std::vector<unsigned int > > > converted;
        for(unsigned int pos = 0; pos < max_len; pos++){
            std::vector<std::vector<unsigned int>> col;
            for(unsigned int f_id = 0; f_id < vec_eos.size(); f_id++){
                std::vector<unsigned int> feat(batch_size, vec_eos.at(f_id));
                for(unsigned int sid = 0; sid < max_batch_size && sid + index < vec_input.size(); sid++){
                    if(pos < vec_input.at(sid + index).size()){
                        feat[sid] = vec_input.at(sid + index).at(pos);
                    }
                }
                col.push_back(feat);
            }
            converted.push_back(col);
        }
        return converted;
    }

    std::vector<std::vector<unsigned int > > align2batch(const unsigned int index, const unsigned int max_batch_size, const std::vector<std::vector<unsigned int> > &vec_input){
        unsigned int max_len = 0;
        unsigned int batch_size = 0;
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < vec_input.size(); sid++){
            batch_size++;
            unsigned int cur_len = vec_input.at(sid + index).size();
            if(cur_len > max_len){
                max_len = cur_len;
            }
        }
        std::vector<std::vector<unsigned int>> col(max_len, std::vector<unsigned int>(batch_size, 0));
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < vec_input.size(); sid++){
            for(unsigned int pos = 0; pos < max_len; pos++){
                if(pos < vec_input.at(sid).size()){
                    col[pos][sid] = vec_input.at(sid + index).at(pos);
                }else{
                    col[pos][sid] = vec_input.at(sid + index).back();
                }
            }
        }
        return col;
    }
};

#endif
