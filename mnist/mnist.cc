#include <torch/torch.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

struct Net: torch::nn::Module{
  Net():
      conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)), 
      conv2(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)), 
      fc1(320, 50), 
      fc2(50, 10){
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv2_drop", conv2_drop);
        register_module("fc1", fc1);
        register_module("fc2", fc1);
  }

  torch::Tensor forward(torch::Tensor x){
    x = conv1->forward(x);
    x = torch::max_pool2d(x, 2);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = conv2_drop->forward(x);
    x = torch::max_pool2d(x, 2);
    x = torch::relu(x);

    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::FeatureDropout conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

struct Options{
  std::string data_root{"data"};
  int32_t batch_size{64};
  int32_t epochs{10};
  double lr{0.01};
  double momentum{0.5};
  bool no_cuda{false};
  int32_t seed{1};
  int32_t test_batch_size{1000};
  int32_t log_interval{10};
};

template<typename DataLoader>
void train(
  int32_t epoch, 
  const Options& options,
  Net& model,
  torch::Device device,
  DataLoader& data_loader,
  torch::optim::SGD& optimizer){
    model.train();//Enables training mode.
    size_t batch_idx = 0;
    data_loader.loop( [&](torch::data::Example<>& batch){
      auto data = batch.data.to(device), labels = batch.label.to(device);
      optimizer.zero_grad();
      auto output = model.forward(data);
      auto loss = torch::nll_loss(output, labels);
      loss.backward();
      optimizer.step();

      if(batch_idx++ % options.log_interval == 0){
        std::cout << "Train Epoch: " << epoch << " ["
                  << batch_idx * batch.data.size(0) << "/"
                  << data_loader.dataset_size() << "]\tLoss: "
                  << loss.toCFloat() << std::endl;
      }
    }

    );
}

template<typename DataLoader>
void test(Net& model, torch::Device device, DataLoader& data_loader){
  torch::NoGradGuard no_grad;
  model.eval();// set eval mode
  double test_loss = 0;
  int32_t correct = 0;
  data_loader.loop([&](torch::data::Example<>& batch){
    auto data = batch.data.to(device), labels = batch.label.to(device);
    auto output = model.forward(data);
    auto cur_loss = torch::nll_loss(output, labels, /*weight=*/{}, 
                                    Reduction::Sum)
    test_loss += cur_loss.toCFloat();

    auto pred = output.argmax(1);
    correct += pred.eq(labels).sum().toCInt();
  });

  test_loss /= data_loader.dataset_size();
  std::cout << "Test set: Average loss: " << test_loss
            << ", Accuracy: " << correct << "/" << data_loader.dataset_size()
            << std::endl;
}

int main(int argc, const char* argv[]){
  torch::manual_seed(0);
  Options options;
  torch::DeviceType device_type;
  if(torch::cuda::is_available() && !options.no_cuda){
    std::cout << "CUDA available! Training on GPU" << std::endl;
    device_type = torch::kCUDA;
  }else{
    std::cout << "Training on CPU" << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);


  Net model;
  model.to(device);
  
  auto train_loader = torch::data::data_loader(
    torch::data::datasets::MNIST(options.data_root, /*train=*/true)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>()),
    torch::data::DataLoaderOptions().batch_size(options.batch_size)
  );

  auto test_loader = torch::data::data_loader(
    torch::data::datasets::MNIST(options.data_root, /*train=*/false)
       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
       .map(torch::data::transforms::Stack<>()),
    torch::data::DataLoaderOptions().batch_size(options.batch_size)
  );

  torch::optim::SGD optimizer(
    model.parameters(),
    torch::optim::SGDOptions(options.lr).momentum(options.momentum)
  );

  for(size_t epoch=1; epoch <= options.epochs;++epoch){
    train(epoch, options, model, device, *train_loader, optimizer);
    test(model, device, *test_loader);
  }
  return 0;
}