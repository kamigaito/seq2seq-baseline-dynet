#include "dynet/dict.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#include "s2s/define.hpp"
#include "s2s/comp.hpp"

#ifndef INCLUDE_GUARD_S2S_CORPORA_UTILS_HPP
#define INCLUDE_GUARD_S2S_CORPORA_UTILS_HPP

namespace s2s {

    void freq_cut_src(const std::string file_path, std::vector<dynet::Dict>& d, std::vector<unsigned int>& word_freq, const std::string unk_symbol, const std::vector<unsigned int>& vec_vocab_size){
        ifstream in(file_path);
        assert(in);
        std::vector<std::map<std::string, unsigned int> > vec_str_freq;
        std::vector<std::vector<std::pair<std::string, unsigned int> > > vec_str_id;
        std::string line;
        while(getline(in, line)) {
            std::vector<std::string>  tokens;
            boost::algorithm::split_regex(tokens, line, boost::regex(" "));
            for(const std::string token : tokens){
                std::vector<std::string> features;
                boost::algorithm::split_regex(features, token, boost::regex("-\\|-"));
                assert(features.size() == d.size());
                if(features.size() != vec_str_freq.size()){
                    vec_str_freq.resize(features.size());
                }
                for(unsigned int feature_id = 0; feature_id < features.size(); feature_id++){
                    vec_str_freq[feature_id][features.at(feature_id)]++;
                }
            }
        }
        in.close();
        for(unsigned int feature_id = 0; feature_id < vec_vocab_size.size(); feature_id++){
            std::vector<std::pair<std::string, unsigned int> > str_vec;
            for(auto& p1: vec_str_freq.at(feature_id)){
                str_vec.push_back(pair<std::string, unsigned int>(p1.first, p1.second));
            }
            CompareString comp;
            sort(str_vec.begin(), str_vec.end(), comp);
            for(auto& p1 : str_vec){
                if(d[feature_id].size() >= vec_vocab_size.at(feature_id) - 1){ // -1 for <UNK>
                    break;
                }
                d[feature_id].convert(p1.first);
            }
        }
        for(unsigned int feature_id = 0; feature_id < vec_vocab_size.size(); feature_id++){
            d[feature_id].freeze(); // no new word types allowed
            d[feature_id].set_unk(unk_symbol);
        }
        word_freq.resize(d.at(0).size());
        for(auto& p1: vec_str_freq.at(0)){
            word_freq[d.at(0).convert(p1.first)] = p1.second;
        }
    }

    void freq_cut_trg(const std::string file_path, dynet::Dict& d, const std::string unk_symbol, unsigned int vocab_size){
        ifstream in(file_path);
        assert(in);
        std::map<std::string, unsigned int> str_freq;
        std::vector<std::pair<std::string, unsigned int> > str_vec;
        std::string line;
        while(getline(in, line)) {
            std::vector<std::string>  tokens;
            boost::algorithm::split_regex(tokens, line, boost::regex(" "));
            for(const std::string token : tokens){
                str_freq[token]++;
            }
        }
        in.close();
        for(auto& p1: str_freq){
            str_vec.push_back(std::pair<std::string, int>(p1.first, p1.second));
        }
        CompareString comp;
        sort(str_vec.begin(), str_vec.end(), comp);
        for(auto& p1 : str_vec){
            if(d.size() >= vocab_size - 1){ // -1 for <UNK>
                break;
            }
            d.convert(p1.first);
        }
        d.freeze(); // no new word types allowed
        d.set_unk(unk_symbol);
    }

    void load_corpus_src(const std::string file_path, const std::vector<unsigned int>& start, const std::vector<unsigned int>& end, std::vector<dynet::Dict>& d, std::vector<std::vector<std::vector<unsigned int> > >& corpus_src){
        ifstream in(file_path);
        assert(in);
        int sid = 0;
        int tlc = 0;
        int ttoks = 0;
        std::string line;
        while(getline(in, line)) {
            ++tlc;
            std::vector<std::string> tokens;
            boost::algorithm::split_regex(tokens, line, boost::regex(" "));
            std::vector<std::vector<unsigned int> > str_tokens(tokens.size(), std::vector<unsigned int>(d.size(), 0));
            ttoks += tokens.size();
            for(unsigned int token_id = 0; token_id < tokens.size(); token_id++) {
                std::vector<std::string> features;
                boost::algorithm::split_regex(features, tokens.at(token_id), boost::regex("-\\|-"));
                for(unsigned int feature_id = 0; feature_id < features.size(); feature_id++){
                    str_tokens[token_id][feature_id] = d[feature_id].convert(features[feature_id]);
                }
            }
            corpus_src.push_back(str_tokens);
            for(unsigned int feature_id = 0; feature_id < start.size(); feature_id++){
                if (corpus_src.back().front().at(feature_id) != start.at(feature_id) && corpus_src.back().back().at(feature_id) != end.at(feature_id)) {
                    std::cerr << "Sentence in " << file_path << ":" << tlc << " didn't start or end with <s>, </s>\n";
                    abort();
                }
            }
            sid++;
        }
        in.close();
        cerr << tlc << " lines, " << ttoks << " tokens, " << d.at(0).size() << " types\n";
    }

    void load_corpus_trg(const std::string file_path, const int start, const int end, dynet::Dict& d, std::vector<std::vector<unsigned int > >& corpus_trg){
        ifstream in(file_path);
        assert(in);
        int tlc = 0;
        int ttoks = 0;
        std::string line = "";
        while(getline(in, line)) {
            ++tlc;
            std::vector<std::string>  tokens;
            std::vector<unsigned int> words;
            boost::algorithm::split_regex(tokens, line, boost::regex(" "));
            for(const std::string token : tokens){
                words.push_back(d.convert(token));
            }
            corpus_trg.push_back(words);
            ttoks += tokens.size();
            if (corpus_trg.back().front() != start && corpus_trg.back().back() != end) {
                std::cerr << "Sentence in " << file_path << ":" << tlc << " didn't start or end with <s>, </s>\n";
                abort();
            }
        }
        in.close();
        cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
    }

    void load_align_corpus(const std::string file_path, std::vector<std::vector<unsigned int> >& corpus){
        ifstream in(file_path);
        assert(in);
        int tlc = 0;
        int ttoks = 0;
        std::string line = "";
        while(getline(in, line)) {
            ++tlc;
            std::vector<std::string> tokens;
            boost::algorithm::split_regex(tokens, line, boost::regex(" "));
            std::vector<unsigned int> aligns(tokens.size());
            for(const std::string token : tokens){
                std::vector<std::string> w2w;
                boost::algorithm::split_regex(w2w, token, boost::regex("-"));
                aligns[stoi(w2w.at(1))] = stoi(w2w.at(0));
            }
            corpus.push_back(aligns);
            ttoks += tokens.size();
        }
        in.close();
        std::cerr << tlc << " lines, " << ttoks << " tokens\n";
    }

    std::vector<std::vector<unsigned int > > trg2batch(const std::vector<unsigned int> sents_order, const unsigned int index, const unsigned int max_batch_size, const std::vector<std::vector<unsigned int> > &vec_input, const unsigned int eos){
        unsigned int max_len = 0;
        unsigned int batch_size = 0;
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < sents_order.size(); sid++){
            batch_size++;
            unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
            if(cur_len > max_len){
                max_len = cur_len;
            }
        }
        std::vector<std::vector<unsigned int>> col(max_len, std::vector<unsigned int>(batch_size, eos));
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < sents_order.size(); sid++){
            for(unsigned int pos = 0; pos < vec_input.at(sents_order.at(sid + index)).size(); pos++){
                col[pos][sid] = vec_input.at(sents_order.at(sid + index)).at(pos);
            }
        }
        return col;
    }

    std::vector<std::vector<std::vector<unsigned int > > > src2batch(const std::vector<unsigned int> sents_order, const unsigned int index, const unsigned int max_batch_size, const std::vector<std::vector<std::vector<unsigned int> > > &vec_input, const std::vector<unsigned int>& vec_eos){
        unsigned int max_len = 0;
        unsigned int batch_size = 0;
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < sents_order.size(); sid++){
            batch_size++;
            unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
            if(cur_len > max_len){
                max_len = cur_len;
            }
        }
        std::vector<std::vector<std::vector<unsigned int > > > converted;
        for(unsigned int pos = 0; pos < max_len; pos++){
            std::vector<std::vector<unsigned int>> col;
            for(unsigned int f_id = 0; f_id < vec_eos.size(); f_id++){
                std::vector<unsigned int> feat(batch_size, vec_eos.at(f_id));
                for(unsigned int sid = 0; sid < batch_size && sid + index < sents_order.size(); sid++){
                    if(pos < vec_input.at(sents_order.at(sid + index)).size()){
                        feat[sid] = vec_input.at(sents_order.at(sid + index)).at(pos).at(f_id);
                        assert(vec_input.at(sents_order.at(sid + index)).at(pos).size() == vec_eos.size());
                    }
                }
                col.push_back(feat);
            }
            assert(col.front().size() == col.back().size());
            assert(col.size() == vec_eos.size());
            converted.push_back(col);
        }
        return converted;
    }

    std::vector<std::vector<unsigned int > > align2batch(const std::vector<unsigned int> sents_order, const unsigned int index, const unsigned int max_batch_size, const std::vector<std::vector<unsigned int> > &vec_input){
        unsigned int max_len = 0;
        unsigned int batch_size = 0;
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < vec_input.size(); sid++){
            batch_size++;
            unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
            if(cur_len > max_len){
                max_len = cur_len;
            }
        }
        std::vector<std::vector<unsigned int>> col(max_len, std::vector<unsigned int>(batch_size, 0));
        for(unsigned int sid = 0; sid < max_batch_size && sid + index < sents_order.size(); sid++){
            for(unsigned int pos = 0; pos < max_len; pos++){
                if(pos < vec_input.at(sents_order.at(sid + index)).size()){
                    col[pos][sid] = vec_input.at(sents_order.at(sid + index)).at(pos);
                }else{
                    col[pos][sid] = vec_input.at(sents_order.at(sid + index)).back();
                }
            }
        }
        return col;
    }

};

#endif
