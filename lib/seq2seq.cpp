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

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/encdec.hpp"
#include "s2s/decode.hpp"
#include "s2s/define.hpp"
#include "s2s/comp.hpp"
#include "s2s/metrics.hpp"
#include "s2s/options.hpp"

namespace s2s {

    void train(const s2s_options &opts){
        // for debug
        s2s::dicts dicts;
        s2s::parallel_corpus para_corp;
        dicts.set(opts);
        // for debug
        para_corp.load(dicts, opts);
        // for debug
        dicts.save(opts);
        // for debug
        dynet::Model model;
        encoder_decoder* encdec = new encoder_decoder(model, &opts);
        dynet::Trainer* trainer = nullptr;
        if(opts.optim == "sgd"){
            trainer = new dynet::SimpleSGDTrainer(model);
        }else if(opts.optim == "momentum_sgd"){
            trainer = new dynet::MomentumSGDTrainer(model);
        }else if(opts.optim == "adagrad"){
            trainer = new dynet::AdagradTrainer(model);
        }else if(opts.optim == "adadelta"){
            trainer = new dynet::AdadeltaTrainer(model);
        }else if(opts.optim == "adam"){
            trainer = new dynet::AdamTrainer(model);
        }else{
            std::cerr << "Trainer does not exist !"<< std::endl;
            assert(false);
        }
        unsigned int epoch = 0;
        while(epoch < opts.epochs){
            // train
            para_corp.shuffle();
            float align_w = opts.guided_alignment_weight;
            batch one_batch;
            while(para_corp.train_batch(one_batch, opts.max_batch_l, dicts)){
                dynet::ComputationGraph cg;
                float loss_att = 0.0;
                float loss_out = 0.0;
                std::vector<dynet::expr::Expression> i_enc = encdec->encoder(one_batch, cg);
                std::vector<dynet::expr::Expression> i_feed{dynet::expr::zeroes(cg, dynet::Dim({opts.rnn_size * 3}, one_batch.trg.at(0).size()))};
                for (unsigned int t = 0; t < one_batch.trg.size() - 1; ++t) {
                    dynet::expr::Expression i_att_t = encdec->decoder_attention(cg, one_batch.trg[t], i_feed[t], i_enc[0]);
                    if(opts.guided_alignment == true){
                        dynet::expr::Expression i_err = sum_batches(pickneglogsoftmax(i_att_t, one_batch.align[t]));
                        loss_att += as_scalar(cg.incremental_forward(i_err));
                        cg.backward(i_err);
                        trainer->update(align_w * 1.0 / double(one_batch.src.size()));
                    }
                    std::vector<dynet::expr::Expression> i_out_t = encdec->decoder_output(cg, i_att_t, i_enc[1]);
                    i_feed.push_back(i_out_t[1]);
                    dynet::expr::Expression i_err = sum_batches(pickneglogsoftmax(i_out_t[0], one_batch.trg[t+1]));
                    //cg.print_graphviz();
                    loss_att += as_scalar(cg.incremental_forward(i_err));
                    cg.backward(i_err);
                    trainer->update(1.0 / double(one_batch.src.size()));
                }
                std::cout << loss_att << "\t" << loss_out << std::endl;
            }
            trainer->update_epoch();
            trainer->status();
            align_w *= opts.guided_alignment_decay;
            epoch++;
            // dev
            encdec->disable_dropout();
            ofstream dev_sents(opts.rootdir + "/dev_" + to_string(epoch) + ".txt");
            while(para_corp.dev_batch(one_batch, opts.max_batch_l, dicts)){
                dynet::ComputationGraph cg;
                std::vector<std::vector<unsigned int> > osent;
                s2s::greedy_decode(one_batch, osent, encdec, cg, dicts, opts);
                dev_sents << s2s::print_sents(osent, dicts);
            }
            dev_sents.close();
            encdec->set_dropout(opts.dropout);
            // save Model
            ofstream model_out(opts.rootdir + "/" + opts.save_file + "_" + to_string(epoch) + ".model");
            boost::archive::text_oarchive model_oa(model_out);
            model_oa << model << *encdec;
            model_out.close();
        }
    }

    void predict(const s2s_options &opts){
        s2s::dicts dicts;
        dicts.load(opts);
        // load model
        dynet::Model model;
        encoder_decoder* encdec = new encoder_decoder(model, &opts);
        ifstream model_in(opts.modelfile);
        boost::archive::text_iarchive model_ia(model_in);
        model_ia >> model >> *encdec;
        model_in.close();
        encdec->disable_dropout();
        // predict
        s2s::parallel_corpus para_corp;
        batch one_batch;
        ofstream predict_sents(opts.rootdir + "/predict.txt");
        while(para_corp.dev_batch(one_batch, opts.max_batch_l, dicts)){
            dynet::ComputationGraph cg;
            std::vector<std::vector<unsigned int> > osent;
            s2s::greedy_decode(one_batch, osent, encdec, cg, dicts, opts);
            predict_sents << s2s::print_sents(osent, dicts);
        }
        predict_sents.close();
    }

};

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    po::options_description bpo("h");
    s2s::s2s_options opts;
    s2s::set_s2s_options(&bpo, &opts);
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, bpo), vm);
    po::notify(vm);
    dynet::initialize(argc, argv);
    if(vm.at("mode").as<std::string>() == "train"){
        s2s::add_s2s_options_train(&vm, &opts);
        s2s::check_s2s_options_train(&vm, opts);
        std::string file_name = opts.rootdir + "/options.txt";
        struct stat st;
        if(stat(opts.rootdir.c_str(), &st) != 0){
            mkdir(opts.rootdir.c_str(), 0775);
        }
        ofstream out(file_name);
        boost::archive::text_oarchive oa(out);
        oa << opts;
        out.close();
        s2s::train(opts);
    }else if(vm.at("mode").as<std::string>() == "predict"){
        ifstream in(opts.rootdir + "/options.txt");
        boost::archive::text_iarchive ia(in);
        ia >> opts;
        in.close();
        s2s::check_s2s_options_predict(&vm, opts);
        s2s::predict(opts);
    }else if(vm.at("mode").as<std::string>() == "test"){

    }else{
        std::cerr << "Mode does not exist !"<< std::endl;
        assert(false);
    }
}
