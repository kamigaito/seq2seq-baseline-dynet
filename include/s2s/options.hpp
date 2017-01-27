#include <boost/algorithm/string.hpp>

namespace s2s {

    class options {
        public:
        //std::string eval_file;
        std::string data_file;
        std::string val_data_file;
        std::string save_file;
        std::string dict_prefix;
        //std::string rootdir;
        unsigned int num_layers;
        unsigned int rnn_size;
        unsigned int att_size;
        bool shared_input;
        std::vector<unsigned int> enc_feature_vec_size;
        std::vector<unsigned int> enc_feature_vocab_size;
        unsigned int dec_word_vec_size;
        unsigned int dec_word_vocab_size;
        bool guided_alignment;
        float guided_alignment_weight;
        float guided_alignment_decay;
        float guided_output_weight;
        float guided_output_decay;
        unsigned int epochs;
        unsigned int start_epoch;
        std::string optim;
        float learning_rate;
        float dropout;
        float lr_decay;
        unsigned int max_batch_l;
        std::string start_symbol;
        std::string end_symbol;
        std::string unk_symbol;
        unsigned int save_every;
        unsigned int print_every;
        unsigned int seed;
        options(){}
    };

    void set_options(boost::program_options::options_description &bpo, options &opts){
        namespace po = boost::program_options;
        bpo.add_options()
        ("rootdir", po::value<std::string>(&(opts.data_file))->requested(), "source train file")
        ("data_file", po::value<std::string>(&(opts.data_file)), "source train file")
        ("val_data_file", po::value<std::string>(&(opts.val_data_file)), "source train file")
        ("save_file_prefix", po::value<std::string>(&(opts.save_file))->default_value("save"), "source train file")
        ("dict_prefix", po::value<std::string>(&(opts.dict_prefix))->default_value("dict_"), "source train file")
        ("num_layers", po::value<unsigned int>(&(opts.num_layers))->default_value(3), "test input")
        ("rnn_size", po::value<unsigned int>(&(opts.rnn_size))->default_value(256), "test input")
        ("att_size", po::value<unsigned int>(&(opts.att_size))->default_value(256), "batch size")
        ("shared_input", po::value<bool>(&(opts.shared_input))->default_value(false), "batch size")
        ("enc_feature_vec_size", po::value<std::string>()->required(), "target train file")
        ("enc_feature_vocab_size", po::value<std::string>()->required(), "target train file")
        ("dec_word_vec_size", po::value<unsigned int>(&(opts.dec_word_vec_size))->required(), "batch size")
        ("dec_word_vocab_size", po::value<unsigned int>(&(opts.dec_word_vocab_size))->required(), "batch size")
        ("guided_alignment", po::value<bool>(&(opts.guided_alignment))->default_value(false), "batch size")
        ("guided_alignment_weight", po::value<float>(&(opts.guided_alignment_weight))->default_value(0.3), "batch size")
        ("guided_alignment_decay", po::value<float>(&(opts.guided_alignment_decay))->default_value(1.0), "batch size")
        ("guided_output_weight", po::value<float>(&(opts.guided_output_weight))->default_value(1.0), "batch size")
        ("guided_output_decay", po::value<float>(&(opts.guided_output_decay))->defalt_value(1.0), "batch size")
        ("epochs", po::value<unsigned int>(&(opts.epochs))->default_value(15), "batch size")
        ("start_epochs", po::value<unsigned int>(&(opts.epochs))->default_value(1), "batch size")
        ("optim", po::value<std::string>(&(opts.optim))->default_value("sgd"), "source train file")
        ("learning_rate", po::value<float>(&(opts.learning_rate))->default_value(1.0), "batch size")
        ("dropout", po::value<float>(&(opts.dropout))->default_value(0.3), "batch size")
        ("lr_decay", po::value<float>(&(opts.lr_decay))->default_value(1.0), "batch size")
        ("max_batch_l", po::value<unsigned int>(&(opts.max_batch_l))->default_value(32), "batch size")
        ("start_symbol", po::value<std::string>(&(opts.start_symbol))->default_value("<s>"), "source train file")
        ("end_symbol", po::value<std::string>(&(opts.end_symbol))->default_value("</s>"), "source train file")
        ("unk_symbol", po::value<std::string>(&(opts.unk_symbol))->default_value("<unk>"), "source train file")
        ("save_every", po::value<unsigned int>(&(opts.save_every))->default_value(1), "batch size")
        ("print_every", po::value<unsigned int>(&(opts.print_every))->default_value(1), "batch size")
        ("seed", po::value<unsigned int>(&(opts.seed))->default_value(0), "batch size");
    }

    void add_options(const boost::program_options::variables_map &vm, options &opts){
        string delim (",");
        std::vector<std::string> vec_str_enc_feature_vec_size;
        boost::split(vec_str_enc_feature_vec_size, vm.at("enc_feature_vec_size").as<unsigned int>(), boost::is_any_of(delim));
        for(auto feature_vec_size : vec_str_enc_feature_vec_size){
            opts.enc_feature_vec_size.push_back(std::stoi(feature_vec_size));
        }
        std::vector<std::string> vec_str_enc_feature_vocab_size;
        boost::split(vec_str_enc_feature_vocab_size, vm.at("enc_feature_vocab_size").as<unsigned int>(), boost::is_any_of(delim));
        for(auto feature_vocab_size : vec_str_enc_feature_vocab_size){
            opts.enc_feature_vocab_size.push_back(std::stoi(feature_vocab_size));
        }
        assert(opts.enc_feature_vocab_size.size() != opts.enc_feature_vec_size.size());
    }

    bool check_options(const boost::program_options::variables_map &vm, const boost::program_options::options_description &opts){
    
    }
}
