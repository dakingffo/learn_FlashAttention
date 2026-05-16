#pragma once

#include <numeric>
#include <cute/tensor.hpp>

namespace FlashAttention::utility {
    template <std::size_t N>
    struct loop;
    template <std::size_t M>
    struct warmup;
    template <typename Loop, typename WarmUp = warmup<3>>
    struct config;
    template <typename Config>
    struct _timer;

    template <std::size_t N, std::size_t M>
    struct _timer<config<loop<N>, warmup<M>>> {
        _timer(const char* name) {
            std::cout << "Timing: " << name << " for " << N << " iterations\n";
            for (std::size_t i = 0; i < N; ++i) {
                CUTE_CHECK_ERROR(cudaEventCreate(&start[i]));
                CUTE_CHECK_ERROR(cudaEventCreate(&stop[i]));
            }
            if constexpr (M == 0) {
                CUTE_CHECK_ERROR(cudaEventRecord(start[0]));
            }
        }

        ~_timer() {
            std::cout << "Average elapsed time: " << std::accumulate(elapsed, elapsed + N, 0.0f) / N << " ms\n";
            for (std::size_t i = 0; i < N; ++i) {
                cudaEventDestroy(start[i]);
                cudaEventDestroy(stop[i]);
            }
        }

        void operator++() {
            if (index < M) {
                if (++index == M) {
                    CUTE_CHECK_ERROR(cudaEventRecord(start[0]));
                }
                return;
            }
            CUTE_CHECK_ERROR(cudaEventRecord(stop[index - M]));
            CUTE_CHECK_ERROR(cudaEventSynchronize(stop[index - M]));
            CUTE_CHECK_ERROR(cudaEventElapsedTime(&elapsed[index - M], start[index - M], stop[index - M]));
            std::cout << "\tElapsed time for iteration " << index - M << ": ";
            std::cout << elapsed[index - M] << " ms\n";
            if (++index < N + M) {
                CUTE_CHECK_ERROR(cudaEventRecord(start[index - M]));
            }
        }

        operator bool() const noexcept {
            return index < N + M;
        }

    private:
        std::size_t index = 0;
        float       elapsed[N];
        cudaEvent_t start[N];
        cudaEvent_t stop[N];
    };

} // namespace utility

#define TIMING(name, config) for (utility::_timer<config> timer(name); timer; ++timer)