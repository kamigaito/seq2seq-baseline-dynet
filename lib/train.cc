#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/fast-lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/encdec/Cho2014.hpp"
#include "s2s/encdec/Sutskever2014.hpp"
#include "s2s/encdec/Bahdanau2014.hpp"
#include "s2s/encdec.hpp"
#include "s2s/decode.hpp"
#include "s2s/define.hpp"
#include "s2s/comp.hpp"
#include "s2s/preprocess.hpp"
#include "s2s/metrics.hpp"

using namespace std;
using namespace dynet;
using namespace dynet::expr;

void print_sent(Sent& osent, dynet::Dict& d_trg){
  for(auto wid : osent){
    string word = d_trg.Convert(wid);
    cout << word;
    if(wid == EOS_TRG){
      break;
    }
    cout << " ";
  }
  cout << endl;
}

void train(boost::program_options::variables_map& vm){
  dynet::Dict d_src, d_trg;
  ParaCorp training, dev;
  vector<Sent > training_src, training_trg, dev_src, dev_trg;
  SOS_SRC = d_src.Convert("<s>");
  EOS_SRC = d_src.Convert("</s>");
  cerr << "Reading source language training text from " << vm.at("path_train_src").as<string>() << "...\n";
  FreqCut(vm.at("path_train_src").as<string>(), d_src, vm.at("src-vocab-size").as<unsigned int>());
  d_src.Freeze(); // no new word types allowed
  d_src.SetUnk("<unk>");
  UNK_SRC = d_src.Convert("<unk>");
  vm.at("src-vocab-size").value() = d_src.size();
  //vm.at("src-vocab-size").as<int>() = d_src.size();
  LoadCorpus(vm.at("path_train_src").as<string>(), SOS_SRC, EOS_SRC, d_src, training_src);

  SOS_TRG = d_trg.Convert("<s>");
  EOS_TRG = d_trg.Convert("</s>");
  cerr << "Reading target language training text from " << vm.at("path_train_trg").as<string>() << "...\n";
  FreqCut(vm.at("path_train_trg").as<string>(), d_trg, vm.at("trg-vocab-size").as<unsigned int>());
  d_trg.Freeze(); // no new word types allowed
  d_trg.SetUnk("<unk>");
  UNK_TRG = d_trg.Convert("<unk>");
  vm.at("trg-vocab-size").value() = d_trg.size();
  //vm.at("trg-vocab-size").as<int>() = d_trg.size();
  LoadCorpus(vm.at("path_train_trg").as<string>(), SOS_TRG, EOS_TRG, d_trg, training_trg);
  cerr << "Writing source dictionary to " << vm.at("path_dict_src").as<string>() << "...\n";
  {
    ofstream out(vm.at("path_dict_src").as<string>());
    boost::archive::text_oarchive oa(out);
    oa << d_src;
    out.close();
  }
  cerr << "Writing target dictionary to " << vm.at("path_dict_trg").as<string>() << "...\n";
  {
    ofstream out(vm.at("path_dict_trg").as<string>());
    boost::archive::text_oarchive oa(out);
    oa << d_trg;
    out.close();
  }
  // for sorting
  for(unsigned int i = 0; i < training_src.size(); i++){
//cerr << i << " " << training_src.at(i).size() << " " << training_trg.at(i).size() << endl;
    ParaSent p(training_src.at(i), training_trg.at(i));
    training.push_back(p);
  }
  cerr << "creating mini-batches" << endl;
  CompareString comp;
  sort(training.begin(), training.end(), comp);
  for(unsigned int i = 0; i < training.size(); i += vm.at("batch-size").as<unsigned int>()){
    unsigned int max_len_src = 0;
    unsigned int max_len_trg = 0;
    for(unsigned int j = 0; j < vm.at("batch-size").as<unsigned int>() && i+j < training.size(); ++j){
      if(max_len_src < training.at(i+j).first.size()){
        max_len_src = training.at(i+j).first.size();
      }
      while(max_len_trg < training.at(i+j).second.size()){
        max_len_trg = training.at(i+j).second.size();
      }
    }
    for(unsigned int j = 0; j < vm.at("batch-size").as<unsigned int>() && i+j < training.size(); ++j){
      while(training.at(i+j).first.size() < max_len_src){ // source padding
        training.at(i+j).first.push_back(EOS_SRC);
      }
      while(training.at(i+j).second.size() < max_len_trg){ // target padding
        training.at(i+j).second.push_back(EOS_TRG);
      }
    }
  }
  cerr << "Reading source development text from " << vm.at("path_dev_src").as<string>() << "...\n";
  LoadCorpus(vm.at("path_dev_src").as<string>(), SOS_SRC, EOS_SRC, d_src, dev_src);
  cerr << "Reading target development text from " << vm.at("path_dev_trg").as<string>() << "...\n";
  LoadCorpus(vm.at("path_dev_trg").as<string>(), SOS_TRG, EOS_TRG, d_trg, dev_trg);
  // for sorting
  for(unsigned int i=0; i < dev_src.size(); i++){
    ParaSent p(dev_src.at(i), dev_trg.at(i));
    dev.push_back(p);
  }
  // creating mini-batches
  sort(dev.begin(), dev.end(), comp);
  for(size_t i = 0; i < dev.size(); i += vm.at("batch-size").as<unsigned int>()){
    for(size_t j = 1; j < vm.at("batch-size").as<unsigned int>() && i+j < dev.size(); ++j){
      while(dev.at(i+j).first.size() < dev.at(i).first.size()){ // source padding
        dev.at(i+j).first.push_back(EOS_SRC);
      }
      while(dev.at(i+j).second.size() < dev.at(i).second.size()){ // target padding
        dev.at(i+j).second.push_back(EOS_TRG);
      }
    }
  }
  
  ostringstream os;
  os << "bilm"
     << '_' << vm.at("depth-layer").as<unsigned int>()
     << '_' << vm.at("dim-input").as<unsigned int>()
     << '_' << vm.at("dim-hidden").as<unsigned int>()
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;
  
  Model model;
  EncoderDecoder<Builder>* encdec;
  //AttentionalEncoderDecoder<Builder> encdec(model, vm);
  switch(vm.at("encdec-type").as<unsigned int>()){
    case __Cho2014__:
    encdec = new Cho2014<Builder>(model, &vm);
    break;
    case __Sutskever2014__:
    encdec = new Sutskever2014<Builder>(model, &vm);
    break;
    case __Bahdanau2014__:
    encdec = new Bahdanau2014<Builder>(model, &vm);
    break;
/*
    case __Luong2015__:
    encdec = new Luong2015<Builder>(model, &vm);
    break;
*/
  }
  Trainer* trainer = nullptr;
  switch(vm.at("trainer").as<unsigned int>()){
    case __SGD__:
    trainer = new SimpleSGDTrainer(&model);
    break;
    case __MomentumSGD__:
    trainer = new MomentumSGDTrainer(&model);
    break;
    case __Adagrad__:
    trainer = new AdagradTrainer(&model);
    break;
    case __Adadelta__:
    trainer = new AdadeltaTrainer(&model);
    break;
    case __RMSprop__:
    trainer = new RmsPropTrainer(&model);
    break;
    case __Adam__:
    trainer = new AdamTrainer(&model);
    break;
  }
  //trainer->eta = vm.at("eta").as<float>();
  trainer->clip_threshold *= vm.at("batch-size").as<unsigned int>();
  trainer->clipping_enabled = 0;
  // Set the start point for each mini-batch of training dataset 
  vector<unsigned> order((training.size()+vm.at("batch-size").as<unsigned int>()-1)/vm.at("batch-size").as<unsigned int>());
  for (unsigned i = 0; i < order.size(); ++i){
    order[i] = i * vm.at("batch-size").as<unsigned int>();
  }

  // Set the start point for each mini-batch of development dataset 
  vector<unsigned> dev_order((dev.size()+vm.at("parallel-dev").as<unsigned int>()-1)/vm.at("parallel-dev").as<unsigned int>());
  for (unsigned i = 0; i < dev_order.size(); ++i){
    dev_order[i] = i * vm.at("parallel-dev").as<unsigned int>();
  }

  unsigned lines = 0;
  while(1) {
    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);
    Timer iteration("completed in");
    for (unsigned si = 0; si < order.size(); ++si) {
      cerr  << "source length=" << training.at(order[si]).first.size() << " target length=" << training.at(order[si]).second.size() << std::endl;
      // build graph for this instance
      double loss = 0;
      unsigned bsize = std::min((unsigned)training.size() - order[si], vm.at("batch-size").as<unsigned int>()); // Batch size
      unsigned remain = bsize;
      while(remain > 0){
        unsigned parallel_size = std::min(remain, vm.at("parallel").as<unsigned int>());
std::cout << (order[si] + bsize - remain) << " " << parallel_size << " : " << (bsize - remain) << "/" << bsize << std::endl;
        if(parallel_size == 0) break;
        ComputationGraph cg;
        vector<Expression> errs;
        Batch sents, osents;
        CorpusToBatch(order[si] + bsize - remain, parallel_size, training, sents, osents);
        encdec->Encoder(sents, cg);
        for (int t = 0; t < osents.size() - 1; ++t) {
          std::vector<Expression> exp_vec = encdec->Decoder(cg, osents[t]);
          Expression i_r_t = exp_vec.at(0);
          Expression i_err = pickneglogsoftmax(i_r_t, osents[t+1]);
          errs.push_back(i_err);
        }
        Expression i_nerr = sum_batches(sum(errs));
        //cg.PrintGraphviz();
        loss += as_scalar(cg.forward());
        cg.backward();
        remain -= parallel_size;
      }
      trainer->update((1.0 / double(bsize)));
      cerr << "E = " << (loss / double(bsize)) << " ppl=" << exp(loss / double(bsize)) << ' ' << std::endl;
      cerr << std::endl;
      //cerr  << "source length=" << training.at(order[si]).first.size() << " target length=" << training.at(order[si]).second.size() << std::endl;
    }
    trainer->update_epoch();
    trainer->status();
    
    double dloss = 0;
    for(unsigned int sid = 0; sid < dev.size(); sid++){
        ComputationGraph cg;
        Sent osent;
        Decode::Greedy<Builder>(dev.at(sid).first, osent, encdec, cg, vm);
        dloss -= f_measure(dev.at(sid).second, osent, d_src, d_trg); // future work : replace to bleu
        cerr << "ref" << endl;
        print_sent(dev.at(sid).second, d_trg);
        cerr << "hyp" << endl;
        print_sent(osent, d_trg);
    }
    if (dloss < best) {
      best = dloss;
      ofstream out(vm.at("path_model").as<string>());
      boost::archive::text_oarchive oa(out);
      oa << model;
      out.close();
    }
    ++lines;
    cerr << "\n***DEV [epoch=" << lines << "] F = " << (0 - dloss / (double)dev.size()) << ' ';
  }
  delete trainer;

}

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description opts("h");
  s2s::add_options(opts);
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);
  dynet::Initialize(argc, argv);
  s2s::options options(vm);
  s2s::train(options);
}
