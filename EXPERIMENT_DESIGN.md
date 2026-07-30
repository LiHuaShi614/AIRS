# AIRS 实验设计

## 1. 研究问题

AIRS 需要验证：在持续测试过程中，系统能否在不使用真实标签的情况下，根据当前 Batch Value 和可用资源选择适配强度，并在有限预算下取得更好的准确率与资源平衡。

RRTTA 是底层适配算法，不是全部实验的研究对象。Inference-only、Light、Medium、Full 表示 AIRS 对底层适配过程施加的四种资源档位。实验仅使用 CIFAR-10-C，预训练模型注册名为 Standard，其骨干网络为 WideResNet-28-10。

## 2. 统一在线协议

每个 Batch 按以下顺序处理：

```text
正常推理
-> 记录当前 Batch 的预测
-> 计算 Batch Value
-> 获取资源状态和预算余额
-> 选择适配档位
-> 执行对应强度的无标签更新
-> 更新后的模型服务后续 Batch
```

当前 Batch 的准确率只使用更新前的正常推理结果。真实标签仅用于离线评价，不参与 Batch Value、档位选择或模型更新。

为避免不同策略因先前消耗随机数的数量不同而获得不同增强，每个 Batch 使用由全局 Batch 索引和实验 seed 唯一确定的随机种子。

## 3. Batch Value

Batch Value 只使用正常推理产生的分类概率，不使用 Teacher-Student 分歧。对样本 i：

```text
r_i = p_i^(1) * (p_i^(1) - p_i^(2)) * (1 - H_i / log(C))
V_t = mean_i(r_i)
```

其中 p_i^(1) 和 p_i^(2) 是最高和次高类别概率，H_i 是预测熵，C 是类别数。V_t 是当前 Batch 的平均可靠度。默认阈值为 0.75、0.82、0.88，分别对应 Inference-only、Light、Medium、Full。所有分量均只使用模型预测，真实标签不参与计算。

## 4. 四个资源档位

预算只计算额外适配成本；所有方法共同需要的正常推理成本不计入适配预算，但会计入总延迟和 GPU 时间。

| 档位 | 归一化成本 | 适配样本比例 | 增强次数 |
|---|---:|---:|---:|
| Inference-only | 0.0 | 0% | 0 |
| Light | 0.2 | 25% | 0 |
| Medium | 0.5 | 50% | 1 |
| Full | 1.0 | 100% | 4 |

除 Inference-only 外，RRTTA 的教师和可靠度路由先处理完整 Batch，再根据档位比例限制最终参与梯度更新的最高可靠样本。比例限制只作用于最后的更新样本截断，不提前缩小教师前向或可靠度路由看到的 Batch。所有档位均不创建或更新 Prototype bank，也不使用 Prototype loss。
Full 档仍遵守 AIRS 的“先推理、后更新”在线协议，因此不会用更新后的模型重新计算当前 Batch 的准确率。实验代码不包含任何准确率目标或结果后处理。

## 5. 预算与资源状态

资源状态 a_t 取值 0 到 1，表示当前 Batch 最多允许使用的适配成本。默认资源轨迹包括：

- steady：始终为 1.0；
- sudden_drop：中途由 1.0 降到 0.2；
- recovery：由 0.2 逐步恢复到 0.5 和 1.0；
- cyclic：按 1.0、0.5、0.2、0.5 周期变化。

Resource Credit Bank 每个 Batch 增加预算额度，低价值 Batch 未使用的额度可以保留，但默认最多累计 4 个 Full 单位。

## 6. 六组实验

### E1：四档资源消耗对比

分别强制使用 Inference-only、Light、Medium、Full。E1 图的 a、b 面板分别比较在线准确率和平均 Batch 延迟；c 面板使用每 Batch 的 Teacher forward calls 与 Backward chunks 之和作为无量纲 Cost Proxy，并按四档最大值归一化为 100%；d 面板分别展示 Teacher calls、Backward chunks、完整 Runtime 和累计 GPU time，各指标独立按四档最大值归一化。CSV 仍保留全部原始资源指标。
准确率面板使用带精确数值的聚焦点图，避免从零开始的柱状图掩盖真实档位差异；原始准确率数值不做任何视觉变换。

### E2：预算敏感性与档位分配

在 20%、40%、60%、80%、100% 适配预算下运行完整 AIRS。比较准确率、延迟、预算利用率、预算余额和四档选择比例。

### E3：动态资源变化

在 steady、sudden_drop、recovery、cyclic 四条资源轨迹下运行 AIRS。观察资源变化后档位是否同步降级或恢复，并比较最终准确率。
四条轨迹均绘制完整时间线；资源上限与实际档位同时使用颜色和线型区分，以支持灰度打印。

### E4：资源调度策略对比

在相同 50% 总预算和 steady 资源下比较：

- AIRS：Value、资源状态和 Resource Credit Bank 联合决策；
- Greedy：尽早使用 Full，直到总预算耗尽；
- Periodic：均匀选择 50% Batch 使用 Full；
- Random：随机选择恰好 50% Batch 使用 Full。

主要比较准确率-GPU 时间关系、P95 延迟、教师调用和反向次数。
AIRS 使用最初的 50% Resource Credit Bank（容量为 4 个 Full 单位）；Greedy、Periodic 和 Random 保留各自原有的 50% Global 总预算分配方式。四种策略使用相同的 Full 档实现。

### E5：Value-Resource 联合决策消融

在 40% Global 总预算和 cyclic 资源下比较完整 AIRS、Value-only、Resource-only 和无联合决策的固定 Medium。四种方法使用相同的累计预算节奏，使准确率差异不再由不同预算交付方式造成。比较准确率、延迟、资源违规次数和档位比例。

### E6：Resource Credit Bank 消融

在 40% 预算和 cyclic 资源下比较：

- Credit bank：未使用预算可跨 Batch 累积；
- Fixed quota：每个 Batch 只有固定额度，未使用部分失效；

比较准确率、预算利用率、最终余额和档位比例。

## 7. 统一评价指标

- 在线推理准确率；
- 平均 Batch 延迟和 P95 Batch 延迟；
- 总 GPU 时间和平均 GPU 时间；
- 峰值显存；
- 教师前向调用次数及处理样本数；
- 反向传播次数和优化器更新次数；
- 四个档位的 Batch 数量及比例；
- 计划预算、实际使用预算、预算利用率和余额；
- 资源违规次数。

## 8. 图表输出

每组实验生成一张 quantitative-grid 论文图：

- E1：档位准确率与资源画像；
- E2：预算-准确率曲线、延迟、预算利用率和档位组成；
- E3：资源轨迹与档位响应时间线；
- E4：准确率-GPU 时间关系和 P95 延迟；
- E5：联合决策消融；
- E6：Credit Bank 消融。

E0 不重新运行模型，只汇总 E2 和 E4，生成 AIRS 总体准确率-资源关系图。所有图同时导出 SVG、PDF、PNG 和 TIFF，SVG 保留可编辑文字。当前默认仅使用一个随机种子，因此图中不绘制不存在的误差条。

## 9. 输出结构

```text
outputs/airs/<suite-id>/
  manifest.json
  E0_all_results.csv
  figures/E0_AIRS_overview.*
  E1/
    E1_summary.csv
    figures/E1_tier_resource_profile.*
    tier_inference/batch_metrics.csv
    tier_inference/per_corruption.csv
    tier_inference/summary.csv
    tier_inference/completed.json
  E2/ ... E6/
```

每个原子配置完成后才写入 completed.json。总命令再次运行时会跳过这些配置，因此实验中断后可以继续运行。
