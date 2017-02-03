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

#include "s2s/encdec.hpp"
#include "s2s/decode.hpp"
#include "s2s/define.hpp"
#include "s2s/comp.hpp"
#include "s2s/preprocess.hpp"
#include "s2s/metrics.hpp"

void train(const options& opts){
    s2s::dicts dicts;
    s2s::parallel_corpus para_corp;
    dicts.set(opts);
    para_corp.load(dicts, opts);
    dynet::Model model;
    encoder_decoder* encdec = new encoder_decoder(model, &opts);
    dynet::Trainer* trainer = nullptr;
    if(opts.optim == "sgd"){
        trainer = new SimpleSGDTrainer(&model);
    }else if(opts.optim == "momentum_sgd"){
        trainer = new MomentumSGDTrainer(&model);
    }else if(opts.optim == "adagrad"){
        trainer = new AdagradTrainer(&model);
    }else if(opts.optim == "adadelta"){
        trainer = new AdadeltaTrainer(&model);
    }else if(opts.optim == "adam"){
        trainer = new AdamTrainer(&model);
    }else{
        std::cerr << "Trainer does not exist !"<< std::endl;
        assert(false);
    }
    unsigned int epoch = 0;
    while(epoch < opts.epochs){
        // train
        para_corp.shuffle();
        float align_w = opts.guided_alignment_weight;
        while(para_corp.train_status()){
            batch one_batch = para_corp.train_batch(opts.max_batch_l);
            ComputationGraph cg;
            float loss_att = 0.0;
            float loss_out = 0.0;
            encdec->encoder(one_batch.src, cg);
            dynet::Expression i_feed;
            for (unsigned int t = 0; t < one_batch.trg.size() - 1; ++t) {
                dynet::Expression i_att_t = encdec->decoder_attention(cg, one_batch.trg[t], i_feed);
                if(opts.guided_alignment == true){
                    dynet::Expression i_err = sum_batches(pickneglogsoftmax(i_att_t, one_batch.align[t]));
                    loss_att += as_scalar(cg.incremental_foward());
                    cg.backward(i_err);
                    trainer->update(align_w * 1.0 / double(one_batch.bsize));
                }
                std::vector<dynet::Expression> i_out_t = encdec->decoder_output(cg, i_att_t);
                dynet::Expression i_err = sum_batches(pickneglogsoftmax(i_out_t[0], one_batch.trg[t+1]));
                i_feed = i_out_t[1];
                //cg.PrintGraphviz();
                loss_att += as_scalar(cg.incremental_foward());
                cg.backward(i_err);
                trainer->update(1.0 / double(one_batch.bsize));
            }
        }
        trainer->update_epoch();
        trainer->status();
        align_w *= opts.guided_alignment_decay;
        epoch++;
        // dev
        while(para_corp.dev_status()){
            batch one_batch = para_corp.dev_batch(opts.max_batch_l);
            Dynet::ComputationGraph cg;
            std::vector<std::vector<unsigned int> > osent;
            s2s::greedy_decode(para_corp.src_dev.at(sid), osent, encdec, cg, opts);
            s2s::print_sents(osent, d_trg);
        }
        // save
        ofstream out(opts.rootdir + "/" + opts.save_file + "_" +to_string(epoch) + ".model");
        boost::archive::text_oarchive oa(out);
        oa << model;
        out.close();
    }
}

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description bpo("h");
  s2s::options options();
  s2s::set_options(bpo, opts);
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, bpo), vm);
  po::notify(vm);
  s2s::add_options(&vm, opts);
  s2s::check_options(&vm, opts);
  dynet::Initialize(argc, argv);
  s2s::options options(vm);
  s2s::train(options);
}
