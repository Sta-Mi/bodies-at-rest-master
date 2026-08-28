# SleepFM 床垫场景迁移

`MattressFusionModel` 将预训练 SleepFM epoch embedding 与床垫侧信息融合，用于个性化睡眠质量回归（也可以通过调整 `output_dim` 用于睡眠分期分类）。支持以下输入：

- `sleepfm`: SleepFM 导出的 PSG embedding；
- `pressure`: 归一化后的压力序列，形状为 `[B,T,1,64,27]`；
- `pose`: PressureNet 输出的 SMPL pose/shape 或关节特征；
- `identity`: `pressure_arcface` 的 256 维 embedding。正式实验应使用 `p_select` 的跨 session 特征，而不是训练 session 的指标。

所有非压力特征均为 `[B,T,D]`。每种模态可以附带 `[B,T]` 布尔 mask，也可以在推理时整种缺失；这样可以对比 SleepFM-only、压力-only、SleepFM+压力、以及完整融合四组消融实验。

```python
from mattress import MattressFusionModel

model = MattressFusionModel(sleepfm_dim=1280, pose_dim=82)
prediction = model(
    {"sleepfm": sleepfm_embedding, "pressure": pressure,
     "pose": smpl_features, "identity": identity_embedding},
    masks={"pressure": pressure_valid},
)
```

## 数据对齐建议

1. 使用稳定的 `(subject_id, session, epoch_start)` 主键，把压力帧聚合到与 PSG 相同的 30 秒 epoch；禁止按数组下标直接拼接。
2. 按受试者划分训练、验证和测试集，避免身份 embedding 或同一晚数据泄漏。
3. 只在训练集拟合连续标签和 pose 特征的标准化参数；压力图保持 `[0,1]`。
4. 主报告采用受试者级 bootstrap 置信区间，并同时报告 MAE、RMSE、Pearson/Spearman；与 SleepFM-only、随机初始化和不含身份特征的模型比较。
5. 身份 embedding 仅作为个性化条件，不把预测出的身份类别当作标签。报告完整模型之外，还应报告移除 identity 的结果，以量化隐私与性能之间的权衡。
