#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <json/json.h>
#include <gflags/gflags.h>

namespace th = torch;
namespace nn = torch::nn;

torch::Tensor gelu(torch::Tensor x){
	// torch.erf  Computes the error function of each element
	return x * 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0)));
}

torch::Tensor swish(torch::Tensor x){
	return x * torch::sigmoid(x);
}

struct BertConfig{
	BertConfig(int vocab_size, int hidden_size, int num_hidden_layers,
			int num_attention_heads, int intermediate_size, 
			std::string hidden_act, float hidden_dropout_prob, 
			float attention_probs_dropout_prob, int max_position_embeddings,
			int type_vocab_size, float initializer_range)
	:vocab_size(vocab_size),
	hidden_size(hidden_size),
	num_hidden_layers(num_hidden_layers),
	num_attention_heads(num_attention_heads),
	intermediate_size(intermediate_size),
	hidden_act(hidden_act),
	attention_probs_dropout_prob(attention_probs_dropout_prob),
	max_position_embeddings(max_position_embeddings),
	type_vocab_size(type_vocab_size),
	initializer_range(initializer_range)
		{
			
		if (hidden_size % num_attention_heads != 0){
			std::cerr<<"The hidden size " << hidden_size 
				<<" is not a multiple of the number of attention heads "
				<< num_attention_heads;
			exit(0);
		}
	}

	explicit BertConfig(const Json::Value& root){
		vocab_size = root["vocab_size"].asInt();
		hidden_size = root["hidden_size"].asInt();
		num_hidden_layers = root["num_hidden_layers"].asInt();
		num_attention_heads = root["num_attention_heads"].asInt();
		hidden_act = root["hidden_act"].asInt();
		intermediate_size = root["intermediate_size"].asInt();
		hidden_dropout_prob = root["hidden_dropout_prob"].asInt();
		max_position_embeddings = root["max_position_embeddings"].asInt();
		type_vocab_size = root["type_vocab_size"].asInt();
		initializer_range = root["initializer_range"].asInt();
	}

	explicit BertConfig(const std::string& filename){
		std::ifstream ifs(filename);
		
		//assert(ifs.is_open());
	
		Json::CharReaderBuilder builder;
		builder["collectCommnets"]=false;
		Json::Value root;
		std::string errs;
		bool ok=Json::parseFromStream(builder, ifs, &root, &errs);
		if (!ok){
			std::cerr << "parse failed \n";
		}

		new (this)BertConfig(root);
	}

	friend std::ostream& operator<<(std::ostream& os, const BertConfig& cfg){
		os << "vocab_size:" <<"\t"<< cfg.vocab_size << "\n"
			<< "hidden_size:" <<"\t"<< cfg.hidden_size << "\n"
			<< "num_hidden_layers:" <<"\t"<< cfg.num_hidden_layers << "\n"
			<< "hidden_size:" <<"\t"<< cfg.hidden_size << "\n"
			<< "num_hidden_layers:" <<"\t"<< cfg.num_hidden_layers << "\n"
			<< "num_attention_heads:" <<"\t"<< cfg.num_attention_heads<< "\n"
			<< "hidden_act:" <<"\t"<< cfg.hidden_act << "\n"
			<< "intermediate_size:" <<"\t"<< cfg.intermediate_size << "\n"
			<< "hidden_dropout_prob:" <<"\t"<< cfg.hidden_dropout_prob << "\n"
			<< "attention_probs_dropout_prob:" <<"\t"<< cfg.attention_probs_dropout_prob << "\n"
			<< "max_position_embeddings:" <<"\t"<< cfg.max_position_embeddings << "\n"
			<< "type_vocab_size:" <<"\t"<< cfg.type_vocab_size << "\n"
			<< "initializer_range:" <<"\t"<< cfg.initializer_range << "\n";

		return os;
	}

	std::string toJsonString(){

	}

	int vocab_size;
	int hidden_size=768;
	int num_hidden_layers=12;
	int num_attention_heads=12;
	int intermediate_size=3072;
	std::string hidden_act="gelu";
	float hidden_dropout_prob=0.1;
	float attention_probs_dropout_prob=0.1;
	int max_position_embeddings=512;
	int type_vocab_size=16;
	float initializer_range=0.02;	

};

struct BERTLayerNorm: nn::Module{
	BERTLayerNorm(const BertConfig& cfg, float var_epsilon=1e-12)
		:var_epsilon(var_epsilon){
		gamma = torch::ones(cfg.hidden_size);
		beta =  torch::zeros(cfg.hidden_size);
		gamma = register_parameter("gamma", gamma); // register
		beta = register_parameter("beta", beta);
	}

	torch::Tensor forward(torch::Tensor x){
		th::Tensor u = x.mean(/*dim*/-1, /*keepdim*/true);
		th::Tensor s = (x-u).pow(2).mean(/*dim*/-1, /*keepdim*/true);
		x = (x-u)/th::sqrt(s + var_epsilon);
		return gamma * x + beta;
	}

	torch::Tensor gamma;
	torch::Tensor beta;
	float var_epsilon;

};

struct BERTEmbeddings: nn::Module{
	BERTEmbeddings(const BertConfig& cfg)
		:word_embeddings(cfg.vocab_size, cfg.hidden_size),
		position_embeddings(cfg.max_position_embeddings, cfg.hidden_size),
		token_type_embeddings(cfg.type_vocab_size, cfg.hidden_size),
		LayerNorm(cfg), dropout(cfg.hidden_dropout_prob)
	{
		
	}

	th::Tensor forward(th::Tensor input_ids, th::Tensor token_type_ids){
		int64_t seq_length = input_ids.size(1);

		th::Tensor position_ids = th::arange(seq_length, 
				th::dtype(th::kInt64).device(input_ids.device()));
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids);
		if(!token_type_ids.defined()){
			token_type_ids = th::zeros_like(input_ids);
		}

		th::Tensor w_emb = word_embeddings->forward(input_ids);
		th::Tensor p_emb = position_embeddings->forward(position_ids);
		th::Tensor t_emb = token_type_embeddings->forward(token_type_ids);

		th::Tensor emb = w_emb + p_emb + t_emb;
		emb = LayerNorm.forward(emb);
		emb = dropout->forward(emb);
		return emb;
	}

	nn::Embedding word_embeddings;
	nn::Embedding position_embeddings;
	nn::Embedding token_type_embeddings;

	BERTLayerNorm LayerNorm;
	nn::Dropout dropout;
	
};

struct BERTSelfAttention: nn::Module{
	BERTSelfAttention(const BertConfig& cfg)
		:num_attention_heads(cfg.num_attention_heads),
		 attention_head_size(cfg.hidden_size / cfg.num_attention_heads),
		 all_head_size(num_attention_heads * attention_head_size),
		 query(cfg.hidden_size, all_head_size),
		 value(cfg.hidden_size, all_head_size),
		 key(cfg.hidden_size, all_head_size),
		 dropout(cfg.attention_probs_dropout_prob){
	}

	th::Tensor transposeForScores(th::Tensor x){
		// x size [a, b, c] => [a, b, nh, hs] => [a, nh, b, hs]
		th::IntList size = x.sizes();
		th::IntList newSize({size[0], size[1], num_attention_heads, attention_head_size});
		x = x.view(newSize);
		x = x.permute({0, 2, 1, 3});
		return x;
	}

	th::Tensor forward(th::Tensor hiddenStates, th::Tensor attentionMask){
		mixedQuery = query->forward(hiddenStates);
		mixedKey = key->forward(hiddenStates);
		mixedValue = value->forward(hiddenStates);

		mixedQuery = transposeForScores(mixedQuery);
		mixedKey = transposeForScores(mixedKey);
		mixedValue = transposeForScores(mixedValue);

		//Take the dot product between "query" and "key" to get the raw
		//attention scores.
		th::Tesnor scores = th::matmul(mixedQuery, mixedKey.transpose(3, 4));
		scores = scores / std::sqrt(attention_head_size);
		//Apply the attention mask is (precomputed for all layers in BertModel
		//forward() function)
		scores = scores + attentionMask;

	}

	int num_attention_heads;
	int attention_head_size;
	int all_head_size;
	nn::Linear query;
	nn::Linear key;
	nn::Linear value;
	nn::Dropout dropout;
};
struct Net: nn::Module{
	Net(){
	}

	torch::Tensor forward(torch::Tensor x){
	}
};


//DEFINE_bool(big_menu, true, "Include 'advanced' options in the menu listing");
DEFINE_string(bert_config_file, "",
			                 "bert config file, path/to/bert_config.json");

int main(int argc, char* argv[]){

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (FLAGS_bert_config_file.empty()){
		std::cout << " no bert_config_file\n";
		return 0;
	}


	torch::Tensor x1 = torch::tensor({0., -1., 10.}, torch::kFloat);
	at::Tensor x2 = at::tensor({0., -1., 10.}, torch::kFloat);
	
	torch::Tensor y1 = torch::erf(x1);
	torch::Tensor y2 = torch::erf(x2);

	//Tensor([0., -0.8427, 1.]);
	std::cout << y1 << "\n"; // Variable[CPUFloatType]
	std::cout << y2 << "\n"; // CPUFloatType

	//std::string cfgFileName("tmp.json");
	//BertConfig cfg(cfgFileName);
	//BertConfig cfg(FLAGS_bert_config_file);
	
	th::Tensor a = th::randn({2, 2});
	th::Tensor b = th::randn({2, 2});

	std::cout << (a-b).pow(2) << "\n";

	th::Tensor x, y=th::rand({3,4});
	std::cout << x.defined()<< " " << y.defined() << std::endl;
	std::cout << y.size(0)<< " " << y.size(1) <<std::endl;
	
	x = th::rand({3,4,5});
	th::IntList l = x.sizes();
	std::cout << l << " "<< l.slice(0, l.size()-1)  << std::endl;
	std::vector<int64_t> vec = l.slice(0, l.size()-1).vec();
	vec.push_back(7);
	vec.push_back(8);
	std::cout << th::IntList(vec) << std::endl;
	return 0;
}
