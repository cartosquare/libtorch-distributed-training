#include <c10d/FileStore.hpp>
#include <c10d/TCPStore.hpp>
#include <c10d/ProcessGroupGloo.hpp>

using namespace c10d;

std::vector<std::string> split(char separator, const std::string& string) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (std::getline(ss, item, separator)) {
    pieces.push_back(std::move(item));
  }
  return pieces;
}

int main() {
    auto master_addr = getenv("MASTER_ADDR");
    auto master_port = atoi(getenv("MASTER_PORT"));
    int rank = atoi(getenv("RANK"));
    int size = atoi(getenv("SIZE"));

    //auto store = c10::make_intrusive<FileStore>("/tmp/test", size);
    std::cout << "master addr: " << master_addr << std::endl;
    std::cout << "master port: " << master_port << std::endl;
    auto store = c10::make_intrusive<TCPStore>(master_addr, master_port, size, rank == 0);

    c10d::ProcessGroupGloo::Options options;
    options.timeout = std::chrono::milliseconds(100000);

    char* ifnameEnv = getenv("GLOO_SOCKET_IFNAME");
    if (ifnameEnv) {
      for (const auto& iface : split(',', ifnameEnv)) {
        options.devices.push_back(
            ::c10d::ProcessGroupGloo::createDeviceForInterface(iface));
      }
    } else {
      // If no hostname is specified, this function looks up
      // the machine's hostname and returns a device instance
      // associated with the address that the hostname resolves to.
      options.devices.push_back(
         ::c10d::ProcessGroupGloo::createDefaultDevice());
    }

    std::cout << "#devices: " << options.devices.size() << std::endl;
    ProcessGroupGloo pg(store, rank, size, options);

    const auto ntensors = 10;
    std::vector<at::Tensor> tensors;
    for(auto i = 0; i < ntensors; ++i) {
        auto x = at::ones({1000, 16 * (i + 1)}, at::TensorOptions(at::CUDA(at::kFloat)));
        tensors.push_back(x);
    }

    // kick off work
    std::vector<c10::intrusive_ptr<ProcessGroup::Work>> pending;
    for (auto i = 0; i < ntensors; ++i) {
        std::vector<at::Tensor> tmp = {tensors[i]};
        pending.push_back(pg.allreduce(tmp));
    }

    // wait for work to complete
    for (auto& work : pending) {
        work->wait();
    }

    for (auto i = 0; i < ntensors; ++i) {
	auto d = tensors[i].cpu();
        std::cout << d.sizes() << std::endl;
        auto data = d.data_ptr<float>();
        std::cout << data[0] << " " << data[1] << " " << data[2] << std::endl;
    }
}
