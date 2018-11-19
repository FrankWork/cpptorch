#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>

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
	}

	static BertConfig fromJsonValue(Json::Value root){
		std::string name = root["name"].asString(); // 实际字段保存在这里
		int age = root["age"].asInt(); // 这是整型，转化是指定类型
		cout << name << " " << age << "\n";
		return *this;		
	}

	static BertConfig fromJsonFile(std::string filename){
		std::ifstream ifs(filename);
		
		//assert(ifs.is_open());
	
		Json::Reader reader;
		Json::Value root;
		if (!reader.parse(ifs, root, false)){
			cerr << "parse failed \n";
			return;
		}

		return fromJsonValue(root);
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

struct Net: torch::nn::Module{
	Net(){
	}

	torch::Tensor forward(torch::Tensor x){
	}
};



int main(){
	torch::Tensor x1 = torch::tensor({0., -1., 10.}, torch::kFloat);
	at::Tensor x2 = at::tensor({0., -1., 10.}, torch::kFloat);
	
	torch::Tensor y1 = torch::erf(x1);
	torch::Tensor y2 = torch::erf(x2);

	//Tensor([0., -0.8427, 1.]);
	std::cout << y1 << "\n"; // Variable[CPUFloatType]
	std::cout << y2 << "\n"; // CPUFloatType

	BertConfig cfg = BertConfig.fromJsonFile("tmp.json");
	

	return 0;
}
