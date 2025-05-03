# ParaGraph

基于 GPU-CPU 协同计算的高性能跨模态索引构建框架

===========================================

# 构建说明 (Build Instructions)

请按照以下步骤编译项目：

1. 创建并进入构建目录：
   mkdir -p build
   cd build
2. 使用 CMake 配置项目（假设 CMakeLists.txt 位于上级目录）：
   cmake ..
3. 使用 Make 编译（-j 可并行加速）：
   make -j
4. 返回项目根目录：
   cd ..

===========================================

# 使用方法 (Usage)

编译成功后，在项目根目录下运行以下脚本启动程序：

```
bash run_t2i.sh
```

===========================================

# 技术细节与注意事项 (Technical Details & Notes)

1. 数据格式 (Data Format)

---

• Ground Truth (gt):

- 需统一 transform 为 128 维
- 若维度 > 128：截断
- 若维度 < 128：不足部分填充 -1

• 输入数据 (data):

- 格式为 [num × dim] 的矩阵
- num 和 dim 需在程序 t2iend 中预定义

2. ParaGraph 配置 (ParaGraph Configuration)

---

图处理流程包括：

• 图一：Top1 Projection（GPU）

- 度数设为 55

• 图二：Search-Refine（GPU）

- 度数设为 64

• 图三：Multi-round Projection（GPU）

- 度数设为 55

• 融合处理（Fusion）：

- 前三张图在 CPU 上融合，生成最终图

• 图四：最终融合图

- 度数设为 70

3. GPU 内存管理 (GPU Memory Management)

---

• 内存预分配：

- GPU 启动时预申请固定内存空间以优化性能

• 固定度数：

- 所有图节点的出度和入度固定为 128

• 空间复用：

- 固定度数策略可实现内存高效复用

• 数据维度关联：

- gt 数据需 resize 为 128 维，以匹配内存结构

===========================================

# 构建流程：

cd build
cmake ..
make -j
cd ..

# 启动程序：

bash run_paragraph.sh

===========================================

# 注意点总结

- gt 数据需通过 transform 转为 128 维（多则截断，少则填 -1）
- 输入数据为 num×dim 格式，需在程序中定义 num 与 dim
- 图处理采用 3 张 GPU 图 + 1 张融合图，度数分别为：
  → 图一 55，图二 64，图三 55，图四（融合）70
- 为实现 GPU 高效执行，需固定图的出入度为 128，空间复用（gt128）
