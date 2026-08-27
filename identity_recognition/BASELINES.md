# 压力分布身份识别：模型与开源实现

## 先说结论

目前没有公认的“床垫压力身份识别”公开排行榜，也没有一个可直接称为该任务 SOTA
的模型。多数压力床垫论文研究姿态、呼吸或离床检测，而不是跨姿态身份识别。因此，严谨的
做法是在相同 SLP 划分上比较若干强视觉表征模型，而不是把普通 ImageNet 分类器直接称为
压力身份识别 SOTA。

本项目推荐的主基线是：

> **压力域自监督预训练（MAE/DINO） + ViT-B/14 或 ViT-B/16 + ArcFace 身份头**

原因是 SLP 只有 101 个可用 subject，直接微调整个大模型容易过拟合；先使用全部无标签
压力帧做掩码重建或自蒸馏，再用角度间隔身份损失微调，更符合小样本生物识别任务。

## 推荐优先级

| 优先级 | 模型方案 | 适合用途 | 官方/主要源码 |
|---|---|---|---|
| 1 | DINOv2 ViT-B/14 + ArcFace | 最强迁移表征候选；建议先做压力域自监督适配 | [DINOv2](https://github.com/facebookresearch/dinov2)、[InsightFace/ArcFace](https://github.com/deepinsight/insightface) |
| 2 | MAE ViT-B/16 + ArcFace 或 SupCon | 用所有压力帧做掩码自监督预训练，适合无额外身份标注的数据 | [MAE](https://github.com/facebookresearch/mae)、[SupContrast](https://github.com/HobbitLong/SupContrast) |
| 3 | ConvNeXt V2 Base + ArcFace | 强卷积基线；FCMAE 预训练思想与稀疏压力图较匹配 | [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2) |
| 4 | Swin Transformer V2 Base + ArcFace | 多尺度局部建模候选，对身体局部接触区域有潜力 | [Swin Transformer](https://github.com/microsoft/Swin-Transformer) |
| 5 | ResNet-18/50 + softmax、SupCon 或 ArcFace | 必须保留的可复现下界，用于检查数据和协议是否正确 | [torchvision](https://github.com/pytorch/vision)、[SupContrast](https://github.com/HobbitLong/SupContrast) |

这些是**值得在压力数据上验证的先进候选**，不是已经在统一压力身份识别排行榜上证明的
名次。DINOv2/MAE 的 RGB 预训练权重可用于初始化，但单通道压力图仍需要压力域自监督
适配；仅复制成三通道并微调，不足以证明其适合该模态。

## 不应只做普通 101 类分类

身份识别至少应报告两种设置：

1. **Closed-set identification**：训练和测试是同一批人，但测试姿态/session 不出现在训练中；
   报告 sample top-1/top-5、每人聚合 top-1 和 macro accuracy。
2. **Verification**：以成对压力图判断是否同一人；报告 ROC-AUC、EER、TAR@FAR，使用
   ArcFace/SupCon 学到的归一化 embedding 和余弦距离。

随机按帧划分会把同一姿态的近重复样本泄漏到训练和测试，得到虚高结果。应优先按
session、睡姿或采集时间分组；如果测试 subject 从未出现在训练中，就不能继续使用固定的
101 类 softmax 评价，而应使用 verification 或 open-set identification。

## 公共数据集

### SLP（当前首选）

- 多模态床上数据，包含压力、深度/RGB 等模态；本仓库清洗后的 subject 列表包含 101 人。
- [SLP 官方项目/数据说明](https://ostadabbas.sites.northeastern.edu/slp-dataset-for-multimodal-in-bed-pose-estimation-3/)
- [BodyMAP 官方源码](https://github.com/RCHI-Lab/BodyMAP) 提供本仓库所用的 SLP 清洗数据组织方式。
- 适合 closed-set 身份实验，但数据最初不是为生物识别设计，发表结果时必须明确重新定义
  的身份协议。

### BodyPressureSD / PressurePose

- [BodyPressure 官方源码与数据说明](https://github.com/Healthcare-Robotics/BodyPressure)
- 主要服务于压力图人体姿态/形状估计，包含合成数据；合成“人物参数”不能自动等价于真实
  生物身份，因此更适合压力域预训练，不应直接作为真实身份识别结论。

## 建议的实验矩阵

在相同 split、输入尺寸和增强策略下至少比较：

1. SmallCNN（只用于数据管线 sanity check；全局池化会丢失压力位置关系）；
2. PressureCNN（保留 `8×4` 空间网格的压力原生监督基线）；
3. ResNet-18 + cross-entropy；
4. ConvNeXt V2 Base + cross-entropy（当前实现）；
5. ConvNeXt V2 Base + ArcFace；
6. MAE 压力域预训练 + ViT-B/16 + ArcFace；
7. DINOv2 ViT-B/14 压力域适配 + ArcFace。

本仓库已提供 `pressure_arcface`：PressureCNN 空间编码器输出 L2-normalized embedding，
训练时使用 ArcFace 角度间隔分类头，验证/推理时使用无 margin 的 cosine logits。评估脚本
会额外保存 `embeddings.pt`，用于后续 verification、模板注册和多帧 embedding 聚合。

除准确率外，同时报告参数量、FLOPs、推理延迟，以及至少 3 个随机种子的均值和标准差。
主结果应来自独立 session/姿态测试，而不是训练样本上的过拟合准确率。

## 实现注意事项

- 保持压力图纵横比，使用 padding 后 resize；不要直接拉伸成人体比例失真的正方形。
- 除归一化压力值外，可加入二值 contact mask、总压力、接触面积和压力质心作为辅助输入。
- 数据增强应符合传感器物理规律：小幅平移、传感器噪声、随机坏点和强度缩放；左右翻转
  是否合理需由床垫方向和实验协议决定。
- 用 ArcFace 时先输出 L2-normalized embedding，再施加角度间隔头；推理时保存 embedding，
  才能同时支持 identification、verification 和后续新用户注册。
