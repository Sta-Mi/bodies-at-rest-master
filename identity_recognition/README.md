# PressurePose 身份识别

本目录使用 `data_BR/real/<subject>/` 中的真实床垫压力记录做身份识别。姿态估计的
`data_BR/convnets` 和 `data_BR/final_results` 不会作为身份标签或预训练权重使用。
合成数据也不用于最终身份测试，因为 PressurePose synth 文件没有真实受试者身份标签。

## 数据协议

默认使用跨 session 协议：

* enrollment/train：每人的 `prescribed.p`；
* probe/test：每人的 `p_select.p`；
* 类别由 `data_BR/real` 下的受试者目录自动发现。

这比随机按帧划分更严格，可避免同一记录中的相邻帧同时进入训练和测试。每个 pickle 的
`images` 字段会被转换为 `1×64×27` 压力图，并按传感器范围归一化到 `[0, 1]`。

## 训练

安装 `requirements.txt` 中的 PyTorch 依赖后，在仓库根目录运行：

```bash
python identity_recognition/train_identity.py \
  --data_root data_BR \
  --model pressure_arcface \
  --train_sessions prescribed \
  --val_sessions p_select \
  --epochs 100 \
  --out_dir identity_recognition/runs/pressure_arcface
```

最佳权重写入 `best_model.pt`，逐 epoch 指标写入 `metrics.jsonl`。训练使用 ArcFace 与
监督对比损失，并以 ROC-AUC、EER、人员级准确率和样本级准确率依次选择最佳 checkpoint。

## 评估

闭集 Top-1/Top-5，并导出 verification embedding：

```bash
python identity_recognition/eval_identity.py \
  --checkpoint identity_recognition/runs/pressure_arcface/best_model.pt \
  --data_root data_BR \
  --out_dir identity_recognition/runs/pressure_arcface/eval
```

计算 ROC-AUC、EER 和 TAR@FAR（10%、1%、0.1%）：

```bash
python identity_recognition/eval_verification.py \
  --embeddings identity_recognition/runs/pressure_arcface/eval/embeddings.pt
```

以训练 session 的平均 embedding 注册模板，再在独立 session 上同时评估识别与验证：

```bash
python identity_recognition/eval_template_identity.py \
  --checkpoint identity_recognition/runs/pressure_arcface/best_model.pt \
  --data_root data_BR \
  --out_dir identity_recognition/runs/pressure_arcface/template_eval
```

正式报告应包含 sample Top-1、Top-5、subject Top-1、ROC-AUC、EER、TAR@FAR=1% 和
TAR@FAR=0.1%。不要用包含训练 session 的 embedding 报告正式 verification 结果。
