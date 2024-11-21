#include <cstddef>
#include <memory>

template <typename T>
class Matrix {
  public:
    std::unique_ptr<T> ptr;
    size_t size;
    Matrix (size_t size);
};

