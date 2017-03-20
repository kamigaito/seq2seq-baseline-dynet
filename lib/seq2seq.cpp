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
#include <time.h>

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
        s2s::dicts dicts;
        s2s::parallel_corpus para_corp_train;
        s2s::parallel_corpus para_corp_dev;
        dicts.set(opts);
        para_corp_train.load_src(opts.srcfile, dicts);
        para_corp_train.load_trg(opts.trgfile, dicts);
        para_corp_train.load_check();
        para_corp_dev.load_src(opts.srcvalfile, dicts);
        para_corp_dev.load_trg(opts.trgvalfile, dicts);
        para_corp_dev.load_check();
        if(opts.guided_alignment == true){
            para_corp_train.load_align(opts.alignfile);
            para_corp_train.load_check_with_align();
            para_corp_dev.load_align(opts.alignvalfile);
            para_corp_dev.load_check_with_align();
        }
        // for debug
        dicts.save(opts);
        // for debug
        dynet::Model model;
        encoder_decoder* encdec = new encoder_decoder(model, &opts);
        encdec->enable_dropout();
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
        float learning_rate = opts.learning_rate;
        trainer->eta0 = learning_rate;
        trainer->eta = learning_rate;
        trainer->eta_decay = 0.f;
        trainer->clipping_enabled = opts.clipping_enabled;
        trainer->clip_threshold = opts.clip_threshold;
        unsigned int epoch = 0;
        float align_w = opts.guided_alignment_weight;
        while(epoch < opts.epochs){
            // train
            para_corp_train.sort_para_sent(opts.sort_sent_type_train, opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train);
            para_corp_train.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
            para_corp_train.shuffle_batch(opts.shuffle_batch_type_train);
            batch one_batch;
            unsigned int bid = 0;
            while(para_corp_train.next_batch_para(one_batch, dicts)){
                bid++;
                one_batch.drop_word(dicts, opts);
                unsigned int batch_size = one_batch.src.at(0).at(0).size();
                //
                auto chrono_start = std::chrono::system_clock::now();
                dynet::ComputationGraph cg;
                std::vector<dynet::expr::Expression> errs_att;
                std::vector<dynet::expr::Expression> errs_out;
                float loss_att = 0.0;
                float loss_out = 0.0;
                std::vector<dynet::expr::Expression> i_enc = encdec->encoder(one_batch, cg);
                std::vector<dynet::expr::Expression> i_feed = encdec->init_feed(one_batch, cg);
                for (unsigned int t = 0; t < one_batch.trg.size() - 1; ++t) {
                    dynet::expr::Expression i_att_t = encdec->decoder_attention(cg, one_batch.trg[t], i_feed[t], i_enc[0]);
                    if(opts.guided_alignment == true){
                        for(unsigned int i = 0; i < one_batch.align.at(t).size(); i++){
                            assert(0 <= one_batch.align.at(t+1).at(i) < one_batch.src.size());
                        }
                        dynet::expr::Expression i_err = pickneglogsoftmax(i_att_t, one_batch.align.at(t+1));
                        errs_att.push_back(i_err);
                    }
                    std::vector<dynet::expr::Expression> i_out_t = encdec->decoder_output(cg, i_att_t, i_enc[1]);
                    i_feed.push_back(i_out_t[1]);
                    dynet::expr::Expression i_err = pickneglogsoftmax(i_out_t[0], one_batch.trg[t+1]);
                    errs_out.push_back(i_err);
                }
                dynet::expr::Expression i_nerr_out = sum_batches(sum(errs_out)) / (float)(batch_size);
                loss_out = as_scalar(cg.forward(i_nerr_out));
                dynet::expr::Expression i_nerr_all;
                if(opts.guided_alignment == true){
                    dynet::expr::Expression i_nerr_att = sum_batches(sum(errs_att)) / (float)(batch_size);
                    loss_att = as_scalar(cg.incremental_forward(i_nerr_att));
                    i_nerr_all = i_nerr_out + align_w * i_nerr_att;
                }else{
                    i_nerr_all = i_nerr_out;
                }
                cg.incremental_forward(i_nerr_all);
                cg.backward(i_nerr_all);
                //cg.print_graphviz();
                trainer->update();
                auto chrono_end = std::chrono::system_clock::now();
                auto time_used = (double)std::chrono::duration_cast<std::chrono::milliseconds>(chrono_end - chrono_start).count() / (double)1000;
                std::cerr << "batch: " << bid;
                std::cerr << ",\tsize: " << batch_size;
                std::cerr << ",\toutput loss: " << loss_out;
                std::cerr << ",\tattention loss: " << loss_att;
                std::cerr << ",\tsource length: " << one_batch.src.size();
                std::cerr << ",\ttarget length: " << one_batch.trg.size();
                std::cerr << ",\ttime: " << time_used << " [s]" << std::endl;
                std::cerr << "[epoch=" << trainer->epoch << " eta=" << trainer->eta << " align_w=" << align_w << " clips=" << trainer->clips_since_status << " updates=" << trainer->updates_since_status << "] " << std::endl;
            }
            para_corp_train.reset_index();
            trainer->update_epoch();
            trainer->status();
            std::cerr << std::endl;
            // dev
            encdec->disable_dropout();
            para_corp_dev.sort_para_sent(opts.sort_sent_type_pred, opts.max_batch_pred, opts.src_tok_lim_pred, opts.trg_tok_lim_pred);
            para_corp_dev.set_para_batch_order(opts.max_batch_pred, opts.src_tok_lim_pred, opts.trg_tok_lim_pred, opts.batch_type_pred);
            std::cerr << "dev" << std::endl;
            std::vector<std::string> str_sents(para_corp_dev.src.size());
            while(para_corp_dev.next_batch_para(one_batch, dicts)){
                std::vector<std::vector<unsigned int> > osent;
                if(opts.decoder_type == "greedy"){
                    s2s::greedy_decode(one_batch, osent, encdec, dicts, opts);
                }else if(opts.decoder_type == "greedy_vinyals"){
                    s2s::greedy_decode_vinyals(one_batch, osent, encdec, dicts, opts);
                }else{
                    std::cerr << "Decoder does not exist !"<< std::endl;
                    assert(false);
                }
                std::vector<std::string> str_batch_sents = s2s::print_sents(osent, dicts);
                for(unsigned int i=0; i < str_batch_sents.size(); i++){
                    // debug
                    std::cerr << one_batch.sent_id.at(i) << std::endl;
                    str_sents[one_batch.sent_id.at(i)] = str_batch_sents.at(i);
                }
            }
            std::string print_body = "";
            for(const std::string str_sent : str_sents){
                print_body += str_sent;
                print_body += "\n";
            }
            ofstream dev_sents(opts.rootdir + "/dev_" + to_string(epoch) + ".txt");
            dev_sents << print_body;
            dev_sents.close();
            para_corp_dev.reset_index();
            encdec->enable_dropout();
            // save Model
            ofstream model_out(opts.rootdir + "/" + opts.save_file + "_" + to_string(epoch) + ".model");
            boost::archive::text_oarchive model_oa(model_out);
            model_oa << model << *encdec;
            model_out.close();
            // preparation for next epoch
            epoch++;
            if(epoch >= opts.sgd_start_epoch){
                if(epoch > opts.sgd_start_decay){
                    if((epoch - opts.sgd_start_decay) % opts.sgd_start_decay_for_each == 0){
                        learning_rate *= opts.sgd_start_lr_decay;
                    }
                }else if(epoch == opts.sgd_start_epoch){
                    delete(trainer);
                    trainer = new dynet::SimpleSGDTrainer(model);
                    learning_rate = opts.sgd_start_learning_rate;
                    trainer->eta0 = learning_rate;
                    trainer->eta_decay = 0.f;
                    trainer->clipping_enabled = opts.clipping_enabled;
                    trainer->clip_threshold = opts.clip_threshold;
                    trainer->epoch = epoch;
                }
            }else{
                if(epoch >= opts.start_epoch){
                    if(epoch > opts.start_epoch){
                        if((epoch - opts.start_epoch) % opts.decay_for_each == 0){
                            learning_rate *= opts.lr_decay;
                        }
                    }else if(epoch == opts.start_epoch){
                        learning_rate *= opts.lr_decay;
                    }
                }
            }
            trainer->eta = learning_rate;
            if(opts.guided_alignment == true){
                if(epoch > opts.guided_alignment_start_epoch){
                    if((epoch - opts.guided_alignment_start_epoch) % opts.guided_alignment_decay_for_each == 0){
                        align_w *= opts.guided_alignment_decay;
                    }
                }else if(epoch == opts.guided_alignment_start_epoch){
                    align_w *= opts.guided_alignment_decay;
                }
            }
        }
    }

    void predict(const s2s_options &opts){
        s2s::dicts dicts;
        dicts.load(opts);
        // load model
        dynet::Model model;
        encoder_decoder* encdec = new encoder_decoder(model, &opts);
        //encdec->disable_dropout();
        ifstream model_in(opts.modelfile);
        boost::archive::text_iarchive model_ia(model_in);
        model_ia >> model >> *encdec;
        model_in.close();
        // predict
        s2s::monoling_corpus mono_corp;
        mono_corp.load_src(opts.srcfile, dicts);
        batch one_batch;
        encdec->disable_dropout();
        mono_corp.sort_mono_sent(opts.sort_sent_type_pred);
        mono_corp.set_mono_batch_order(opts.max_batch_pred, opts.src_tok_lim_pred, opts.batch_type_pred);
        std::vector<std::string> str_sents(mono_corp.src.size());
        while(mono_corp.next_batch_mono(one_batch, dicts)){
            std::vector<std::vector<unsigned int> > osent;
            if(opts.decoder_type == "greedy"){
                s2s::greedy_decode(one_batch, osent, encdec, dicts, opts);
            }else if(opts.decoder_type == "greedy_vinyals"){
                s2s::greedy_decode_vinyals(one_batch, osent, encdec, dicts, opts);
            }else{
                std::cerr << "Decoder does not exist !"<< std::endl;
                assert(false);
            }
            std::vector<std::string> str_batch_sents = s2s::print_sents(osent, dicts);
            for(unsigned int i=0; i < str_batch_sents.size(); i++){
                str_sents[one_batch.sent_id.at(i)] = str_batch_sents.at(i);
            }
        }
        std::string print_body = "";
        for(const std::string str_sent : str_sents){
            print_body += str_sent;
            print_body += "\n";
        }
        ofstream pred_sents(opts.trgfile);
        pred_sents << print_body;
        pred_sents.close();
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
        dynet::initialize(argc, argv);
        s2s::train(opts);
    }else if(vm.at("mode").as<std::string>() == "predict"){
        ifstream in(opts.rootdir + "/options.txt");
        boost::archive::text_iarchive ia(in);
        ia >> opts;
        in.close();
        s2s::check_s2s_options_predict(&vm, opts);
        dynet::initialize(argc, argv);
        s2s::predict(opts);
    }else if(vm.at("mode").as<std::string>() == "test"){

    }else{
        std::cerr << "Mode does not exist !"<< std::endl;
        assert(false);
    }
}
