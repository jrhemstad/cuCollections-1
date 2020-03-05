#include <cuda/std/atomic>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <iostream>


template <typename F, typename S>
struct alignas(8) my_pair{
    F first;
    S second;

    template <typename Other>
    __host__ __device__
    my_pair& operator=(Other const&){
        printf("assignment\n");
    }
};

template <typename F, typename S>
__host__ __device__
 my_pair<F,S> make_my_pair(F f, S s){
    return my_pair<F,S>{f,s};
}

__global__ void example(cuda::atomic<my_pair<int,int>, cuda::thread_scope_device> * a) {
    auto expected = make_my_pair(0,0);
    a->compare_exchange_strong(expected, make_my_pair(1,1));
}

int main(){
    using pair_type = my_pair<int, int>;
    std::cout << "alignof pair_type: " << alignof(pair_type) << std::endl
              << "size of pair_type: " << sizeof(pair_type) << std::endl;

    using atomic_pair = cuda::atomic<pair_type, cuda::thread_scope_device>;
    thrust::device_vector<atomic_pair> a(1);
    example<<<1,1>>>(a.data().get());
    if(cudaSuccess != cudaDeviceSynchronize()){
        std::cout << "error\n";
    }
}
