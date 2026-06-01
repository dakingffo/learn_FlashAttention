# learn_FlashAttention (CuTe Learning Project)

这是一个基于 C++/CUDA 和 CuTe 库从零手写的 FlashAttention-2 学习项目，包含了完整的前向（Forward）和反向（Backward）传播内核实现。

本项目主要用于深入学习 GPU 底层架构、CuTe Layout 几何代数映射以及高性能算子开发，不以通用开源库的工程化封装为目的。


## FlashAttentionV2

* **基础设施**：C++ / CUDA / CUTLASS / CuTe
* **并行计算与流水线**：
    * 基于 `cp.async` 的多阶段（Multi-stage）Global-to-Shared 异步拷贝流水线。
    * 非对称 Tile 设计与双 Traits (Dual-Traits) 静态分发架构，适配前反向不同的算术强度。
* **寄存器与访存优化**：
    * 利用 CuTe Permutation 机制实现多 Warp 级别的寄存器数据复用（C to A Layout Remapping）。
    * Shared Memory Swizzling 规避 Bank Conflict。
* **鲁棒性与泛化**：
    * 支持任意序列长度（`seq_len`）的越界安全处理（Identity 坐标映射与掩码填充）。
    * 支持 FP16 / BF16 数据格式，以及 HeadDim 64/128 配置。
  
```text
├── FlashAttentionV2/
│   ├── main.cpp                 # 测试入口：精度校验与性能 benchmark (PyTorch Naive vs CuTe)
│   ├── traits.hpp               # 核心静态分发配置：定义 FWD/BWD 的 Tile 大小、MMA 刀模与 Smem Layout
│   ├── forward_launch.cu        # 前向传播 Host 端 Launch 逻辑
│   ├── forward_kernel.hpp       # 前向传播 Device 端 Kernel 实现 (Online Softmax)
│   ├── backward_launch.cu       # 反向传播 Host 端 Launch 逻辑
│   ├── backward_kernel.hpp      # 反向传播 Device 端 Kernel 实现 (dQ, dK, dV)
│   ├── boundary_algorithm.hpp   # 跨边界内存读取与越界填充 (Masking & Padding) 逻辑
|   └── CMakeLists.txt           # CMake子目录
└── CMakeLists.txt               # 编译配置
```
## FlashAttentionV3
**comming soon...**

