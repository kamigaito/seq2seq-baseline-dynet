
class dict {
    public:
    dynet::Dict d_src
    dynet::Dict d_trg;
    unsigned int source_start_id;
    unsigned int source_end_id;
    unsigned int source_unk_id;
    unsigned int target_start_id;
    unsigned int target_end_id;
    unsigned int target_unk_id;
    void set(const options &opts){
        source_start_id = d_src.Convert(opts.start_symbol);
        source_end_id = d_src.Convert(opts.end_symbol);
        cerr << "Reading source language training text from " << vm.at("path_train_src").as<string>() << "...\n";
        freq_cut(vm.at("path_train_src").as<string>(), d_src, vm.at("src-vocab-size").as<unsigned int>());
        d_src.Freeze(); // no new word types allowed
        d_src.SetUnk(opts.unk_symbol);
        source_unk_id = d_src.Convert(opts.unk_symbol);
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
    }
    void load(){
    }
    void save(){
    }
}

class parallel_corpus {
    std::vector<std::vector<unsigned int> > source;
    std::vector<std::vector<unsigned int> > target;
    std::vector<std::vector<unsigned int> > alignment;
    set(const dict &dic){
    
    }
}
