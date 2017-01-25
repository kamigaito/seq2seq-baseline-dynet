
namespace s2s {

    template <typename real>
    class options {
        public:
        std::string data_file;
        std::string val_data_file;
        std::string savefile;
        std::string dict_prefix;
        std::string rootdir;
        std::string num_shards;
        unsigned int train_from;
        unsigned int num_layers;
        unsigned int rnn_size;
        unsigned int dec_word_vec_size;
        std::vector<unsigned int> enc_feature_vec_size;
        bool attn;
        bool brnn;
        bool use_chars_enc;
        bool use_chars_dec;
        bool reverse_src;
        bool init_dec;
        bool input_feed;
        bool multi_attn;
        bool res_net;
        bool guided_alignment;
        real guided_alignment_weight;
        real guided_alignment_decay;
        bool char_vec_size;
        unsigned int kernel_width;
        unsigned int num_kernels;
        unsigned int num_highway_layers;
        unsigned int epochs;
        unsigned int start_epoch;
        bool param_init;
        std::string optim;
        real learning_rate;
        real max_grad_norm;
        real dropout;
        real lr_decay;
        bool curriculum;
        bool pre_word_vecs_enc;
        bool pre_word_vecs_dec;
        bool fix_word_vecs_enc;
        bool fix_word_vecs_dec;
        unsigned int max_batch_l;
        std::string start_symbol;
        std::string end_symbol;
        std::string unk_symbol;
        unsigned int save_every;
        unsigned int print_every;
        unsigned int seed;
        options(const boost::program_options::variables_map &vm){
            
        }
    };

    void add_options(boost::program_options::options_description &opts){
        opts.add_options()
        ("path_train_src", po::value<string>()->required(), "source train file")
        ("path_train_trg", po::value<string>()->required(), "target train file")
        ("path_dev_src", po::value<string>()->required(), "source dev file")
        ("path_dev_trg", po::value<string>()->required(), "target dev file")
        ("path_test_src", po::value<string>()->required(), "test input")
        ("path_test_out", po::value<string>()->required(), "test input")
        ("path_dict_src", po::value<string>()->required(), "source dictionary file")
        ("path_dict_trg", po::value<string>()->required(), "target dictionary file")
        ("path_model", po::value<string>()->required(), "test input")
        ("batch-size",po::value<unsigned int>()->default_value(1), "batch size")
        ("parallel",po::value<unsigned int>()->default_value(1), "parallel size")
        ("parallel-dev",po::value<unsigned int>()->default_value(1), "parallel size (dev)")
        ("beam-size", po::value<unsigned int>()->default_value(1), "beam size")
        ("src-vocab-size", po::value<unsigned int>()->default_value(20000), "source vocab size")
        ("trg-vocab-size", po::value<unsigned int>()->default_value(20000), "target vocab size")
        ("builder", po::value<unsigned int>()->default_value(0), "select builder (0:LSTM (default), 1:Fast-LSTM, 2:GRU, 3:RNN)")
        ("trainer", po::value<unsigned int>()->default_value(0), "select trainer (0:SGD (default), 1:MomentumSGD, 2:Adagrad, 3:Adadelta, 4:RMSprop, 5:Adam)")
        ("encdec-type", po::value<unsigned int>()->default_value(2), "select a type of encoder-decoder (0:dynet example, 1:encoder-decoder, 2:attention (default))")
        ("train", po::value<unsigned int>()->default_value(1), "is training ? (1:Yes,0:No)")
        ("test", po::value<unsigned int>()->default_value(1), "is test ? (1:Yes, 0:No)")
        ("bidirection", po::value<unsigned int>()->default_value(1), "use bidirectional encoding ? (1:yes, 0:no)")
        ("attention", po::value<unsigned int>()->default_value(1), "use attention ? (1:yes, 0:no)")
        ("dim-input", po::value<unsigned int>()->default_value(500), "dimmension size of embedding layer")
        ("dim-hidden", po::value<unsigned int>()->default_value(500), "dimmension size of hidden layer")
        ("dim-attention", po::value<unsigned int>()->default_value(64), "dimmension size of hidden layer")
        ("depth-layer", po::value<unsigned int>()->default_value(1), "depth of hidden layer")
        ("length-limit", po::value<unsigned int>()->default_value(100), "length limit of target language in decoding")
        ("eta", po::value<float>()->default_value(1.0), "learning rate")
        ("dynet-mem", po::value<string>()->default_value("512m"), "memory size");
    }

}
