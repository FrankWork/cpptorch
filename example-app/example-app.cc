#include <iostream>

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

int main(){
  at::Tensor a=at::ones({2,2}, at::kInt);
  at::Tensor b=at::randn({2,2});
  auto c=a+b.to(at::kInt);
  std::cout << c << "\n\n";

  at::Tensor tensor = torch::rand({2,3});
  std::cout << tensor << "\n\n";

  at::Tensor d = torch::ones({2,2}, at::requires_grad());
  at::Tensor e = torch::randn({2,2});
  auto f = d+e;
  f.backward();
  std:: cout << d.grad() << "\n\n";

  std::cout << std::endl;
}