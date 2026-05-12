# Grill Session: HIL-SERL 真机部署计划

**日期**: 2025-05-11
**状态**: 进行中，明天继续

## 已确认的决策

### 任务定义
- **任务**: 电机轴轴孔装配（peg-in-hole）
- **三种轴径**: 6mm, 7mm, 8mm，长度 ~71mm
- **7mm 轴特殊结构**: 前 17mm 直径 6.2mm（导向段），后段 7mm（紧配合）
- **配合类型**: 三种均为紧配合（过盈）
- **装配方向**: 竖直从上往下插
- **插入深度要求**: > 20mm
- **插入力**: ~8-10N（Isaac 仿真数据，真机需实测验证）
- **训练顺序**: 先 7mm（有导向段最简单）→ 后续靠视觉泛化到 6/8mm

### 硬件配置

| | CR5AF | Marvin M6 CCS |
|---|---|---|
| 自由度 | 6-DOF | 7-DOF |
| SDK | TCP-IP-Python-V4 (TCP 29999+30004) | TJ_FX_ROBOT_CONTRL_SDK (1kHz UDP) |
| 力传感器 | SixForceValue via RT port | fb_joint_them 复用字段 / 10000 |
| 相机 | 2x D405 RealSense | 2x D405 RealSense |
| 夹爪 | 后续到货 | 后续到货 |
| SpaceMouse | 后续到货 | 后续到货 |
| 优先级 | **先行** | Marvin 坏了，延后 |

### 训练架构
- **训练机**: 本地 2x RTX 5090 (32GB)
- **推理机**: Jetson Thor 128GB (ssh thor)，已接 2x D405
- **网络**: 千兆交换机
- **模式**: Actor-Learner 分离，RPC 通信（与 HIL-SERL 一致）
- **机器人**: 轮流使用，各自独立训练

### 软件设计
- **控制模式**: 笛卡尔阻抗控制 + delta pose 指令（与 HIL-SERL 一致）
- **夹爪支持**: 代码支持两种模式 — 固定法兰（6D action）和夹爪（7D action）
- **SpaceMouse**: 示教采集 demo + HGDagger 人机干预（与 HIL-SERL 一致）
- **奖励函数**: 视觉分类器（先训练分类器 → 再采集 demo → RL 训练）
- **分类器采集**: 遥操作机器人插入采集 + 手动摆位补充数据量

### 代码分支进展
- `feat/cr5af-robot-integration`: cr5af_server.py 完成（Flask API, TCP 协议, FC 力控）
- `feat/marvin-m6-integration`: marvin_server.py + marvin_env.py 完成（Flask API, 阻抗控制, gym 环境）

---

## 待解决问题（自行调查中）

### P0 — CR5AF 控制模式适配
**问题**: CR5AF 没有原生笛卡尔阻抗控制，FC 模式与 HIL-SERL 的阻抗控制行为不同。
- HIL-SERL Franka: 连续发 delta pose → 阻抗控制器执行（弹簧-阻尼模型）
- CR5AF: FC 力控模式是离散的，不支持连续 delta pose 同时保持力柔顺
- **需要研究**: FCForceMode 设 target force=0 + 设置 stiffness/damping 是否能模拟阻抗行为？
- **备选方案**: 软件层实现阻抗仿真（读力反馈 → 计算位置调整 → MovL 执行）

### P1 — 代码缺陷（已确认）
- Marvin `vel` 全是零（marvin_server.py:231），需要用有限差分或 Jacobian 实现
- 力传感器单位未真机验证（CR5AF: SixForceValue 原始单位? Marvin: fb_joint_them/10000 单位?）

### P2 — 待明天确认
- [ ] CR5AF 的工作空间和安全性边界参数
- [ ] Jetson Thor 上软件环境状态（驱动、pyrealsense2、网络配置）
- [ ] D405 相机在 Thor 上的序列号和裁剪参数
- [ ] 分类器训练的具体数据量规划

---

## CR5AF SDK 力控 API 参考

```
ForceDriveMode(x,y,z,rx,ry,rz)     # 启用/禁用各轴力驱动拖拽
ForceDriveSpeed(speed)              # 力驱动速度 [1,100]
FCForceMode(x,y,z,rx,ry,rz,        # 力控模式
  fx,fy,fz,frx,fry,frz)            # 目标力: 平移[-200,200]N, 旋转[-12,12]N·m
FCSetStiffness(x,y,z,rx,ry,rz)     # 刚度系数
FCSetDamping(x,y,z,rx,ry,rz)       # 阻尼系数
FCSetMass(x,y,z,rx,ry,rz)          # 惯性系数
FCSetDeviation(x,y,z,rx,ry,rz)     # 位置/容差限制
FCSetForceLimit(x,y,z,rx,ry,rz)    # 最大力限制
FCSetForceSpeedLimit(x,y,z,rx,ry,rz) # 力控速度限制
FCSetForce(fx,fy,fz,frx,fry,frz)   # 实时调整目标力
FCOff()                             # 退出力控模式
```
