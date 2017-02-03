#include "dynet/dict.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/define.hpp"
#include "s2s/comp.hpp"

#ifndef INCLUDE_GUARD_PREPROCESS_HPP
#define INCLUDE_GUARD_PREPROCESS_HPP

namespace s2s {

    void freq_cut_src(const std::string file_path, std::vector<dynet::Dict>& d, const std::string unk_symbol, std::vector<unsigned int>& vec_vocab_size){
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
                boost::algorithm::split_regex(features, token, boost::regex("-|-"));
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
                if(d.size() >= vec_vocab_size.at(feature_id) - 1){ // -1 for <UNK>
                    break;
                }
                d.Convert(p1.first);
            }
        }
        for(unsigned int feature_id = 0; feature_id < vec_vocab_size.size(); feature_id++){
            d[feature_id].Freeze(); // no new word types allowed
            d[feature_id].SetUnk(unk_symbol);
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
            d.Convert(p1.first);
        }
        d.Freeze(); // no new word types allowed
        d.SetUnk(unk_symbol);
    }

    void load_src_corpus(const std::string file_path, const std::vector<unsigned int>& start, const std::vector<unsigned int>& end, const std::vector<dynet::Dict>& d, std::vector<std::vector<std::vector<unsigned int> > >& corpus){
        ifstream in(file_path);
        assert(in);
        int tlc = 0;
        int ttoks = 0;
        std::string line;
        while(getline(in, line)) {
            ++tlc;
            std::vector<std::string> tokens;
            boost::algorithm::split_regex(tokens, line, boost::regex(" "));
            std::vector<std::vector<unsigned int> > str_tokens(tokens.size(), std::vector<unsigned int>(d.d_src.size(), 0));
            ttoks += tokens.size();
            for(unsigned int token_id = 0; token_id < tokens.size(); token_id++) {
                std::vector<std::string> features;
                boost::algorithm::split_regex(features, tokens.at(token_id), boost::regex("-|-"));
                for(unsigned int feature_id = 0; feature_id < features.size(); feature_id++){
                    str_tokens[token_id][feature_id] = d.at(feature_id).convert(features.at(feature_id));
                }
            }
            corpus.push_back(str_tokens);
            for(unsigned int feature_id = 0; feature_id < start.size(); feature_id++){
                if (corpus.back().at(feature_id).front() != start.at(feature_id) && corpus.back().at(feature_id).front() != end.at(feature_id)) {
                    cerr << "Sentence in " << file_path << ":" << tlc << " didn't start or end with <s>, </s>\n";
                    abort();
                }
            }
        }
        in.close();
        cerr << tlc << " lines, " << ttoks << " tokens, " << d.at(0).size() << " types\n";
    }

    void load_trg_corpus(const std::string file_path, const int start, const int end, dynet::Dict& d, std::vector<Sent>& corpus){
        ifstream in(file_path);
        assert(in);
        int tlc = 0;
        std::string line = "";
        while(getline(in, line)) {
            ++tlc;
            std::vector<std::string>  tokens;
            Sent words;
            boost::algorithm::split_regex(tokens, line, boost::regex(" "));
            for(const std::string token : tokens){
                words.push_back(d.convert(token));
            }
            corpus.push_back(words);
            ttoks += tokens.size();
            if (corpus.back().front() != start && corpus.back().back() != end) {
                cerr << "Sentence in " << file_path << ":" << tlc << " didn't start or end with <s>, </s>\n";
                abort();
            }
        }
        in.close();
        cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
    }

    void load_align_corpus(const std::string file_path, std::vector<std::vector<std::vector<unsigned int> > >& corpus){
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
        cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
    }

};

#endif
