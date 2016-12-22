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

using namespace std;
using namespace dynet;
using namespace dynet::expr;

// return the current word size
void FreqCut(const string file_path, dynet::Dict& d, unsigned int dim_size){
  ifstream in(file_path);
  assert(in);
  map<string, unsigned int> str_freq;
  vector<pair<string, unsigned int> > str_vec;
  string line;
  while(getline(in, line)) {
    std::istringstream words(line);
    while(words){
      string word;
      words >> word;
      str_freq[word]++;
    }
  }
  in.close();
  for(auto& p1: str_freq){
   str_vec.push_back(pair<string, int>(p1.first, p1.second));
  }
  CompareString comp;
  sort(str_vec.begin(), str_vec.end(), comp);
  for(auto& p1 : str_vec){
    if(d.size() >= dim_size - 1){ // -1 for <UNK>
      break;
    }
    d.Convert(p1.first);
  }
}

void LoadCorpus(const string file_path, const int start, const int end, dynet::Dict& d, vector<Sent>& corpus){
  ifstream in(file_path);
  assert(in);
  int tlc = 0;
  int ttoks = 0;
  string line;
  while(getline(in, line)) {
    ++tlc;
    corpus.push_back(ReadSentence(line, &d));
    ttoks += corpus.back().size();
    if (corpus.back().front() != start && corpus.back().back() != end) {
      cerr << "Sentence in " << file_path << ":" << tlc << " didn't start or end with <s>, </s>\n";
      abort();
    }
  }
  in.close();
  cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
}

void CorpusToBatch(const unsigned int bid, const unsigned int bsize, ParaCorp& sents, Batch& lbatch, Batch& rbatch){
  lbatch.resize(sents.at(bid).first.size());
  rbatch.resize(sents.at(bid).second.size());
  for(unsigned int sid = bid; sid< bid + bsize; sid++){
    for(unsigned int i=0; i<sents.at(bid).first.size(); i++){
      lbatch[i].push_back(sents.at(sid).first.at(i));
    }
    for(unsigned int i=0; i<sents.at(bid).second.size(); i++){
      rbatch[i].push_back(sents.at(sid).second.at(i));
    }
  }
}

void CorpusToBatch(const unsigned int bid, const unsigned int bsize, SentList& sents, Batch& batch){
  batch.resize(sents.at(bid).size());
  for(unsigned int sid = bid; sid< bid + bsize; sid++){
    for(unsigned int i=0; i<sents.at(bid).size(); i++){
      batch[i].push_back(sents.at(sid).at(i));
    }
  }
}

void SentToBatch(const Sent& sent, Batch& batch){
	batch.resize(sent.size());
  for(unsigned int sid=0; sid < sent.size(); sid++){
    batch[sid].push_back(sent.at(sid));
	}
}

#endif
