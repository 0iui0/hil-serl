# Grill Session: HIL-SERL 真机部署计划

**日期**: 2025-05-12
**分支**: `feat/motor-assembly-rl`
**状态**: CR5AF + Marvin 已合并，等待真机上电验证

---

## 已确认的决策

### 任务定义
- **当前任务**: 电机轴轴孔装配（peg-in-hole / motor shaft assembly）
- **三种轴径**: 6mm, 7mm, 8mm，长度 ~71mm
- **7mm 轴特殊结构**: 前 17mm 直径 6.2mm（导向段），后段 7mm（紧配合）
- **配合类型**: 三种均为紧配合（过盈）
- **装配方向**: 竖直从上往下插
- **插入深度要求**: > 20mm
- **插入力**: ~8-10N（Isaac 仿真数据，真机需实测验证）
- **训练顺序**: 先 7mm（有导向段最简单）→ 后续靠视觉泛化到 6/8mm
- **远期 6 个装配任务**:
  1. 轴孔装配（shaft insertion）← 当前
  2. 壳体涂胶（housing glue）
  3. 压装（press fitting）
  4. 轴→端子插入（shaft→terminal insertion）
  5. 轴+端子→壳体（shaft+terminal→housing）
  6. 压装+粘接（press+bonding）

### 硬件配置

| | CR5AF | Marvin M6 CCS |
|---|---|---|
| 自由度 | 6-DOF | 7-DOF |
| SDK | TCP-IP-Python-V4 (TCP 29999+30004) | TJ_FX_ROBOT_CONTRL_SDK (1kHz UDP) |
| 力传感器 | SixForceValue via RT port 30004 | fb_joint_them 复用字段 / 10000 |
| 相机 | 2x D405 RealSense | 2x D405 RealSense |
| 夹爪 | 后续到货 | 后续到货 |
| SpaceMouse | 后续到货 | 后续到货 |
| 优先级 | **先行** | 坏了，延后 |

### 训练架构
- **训练机**: 本地 2x RTX 5090 (32GB)
- **推理机**: Jetson Thor 128GB (ssh thor)，已接 2x D405
- **网络**: 千兆交换机
- **模式**: Actor-Learner 分离，RPC 通信（与 HIL-SERL 一致）
- **机器人**: 轮流使用，各自独立训练

### 软件设计
- **控制模式**: 笛卡尔阻抗控制 + delta pose 指令（与 HIL-SERL 一致）
- **夹爪支持**: 代码支持两种模式 — 固定法兰（6D action）和夹爪（7D action），通过 `USE_GRIPPER` 配置切换
- **SpaceMouse**: 示教采集 demo + HGDagger 人机干预（与 HIL-SERL 一致）
- **奖励函数**: 视觉分类器（先训练分类器 → 再采集 demo → RL 训练）
- **分类器采集**: 遥操作机器人插入采集 + 手动摆位补充数据量
- **固定法兰模式**: `GripperCloseEnv` 屏蔽夹爪动作， shaft 预先固定
- **夹爪模式**: reset 时自动抓取 shaft，episode 中控制插入+释放

### 代码分支进展
- `feat/motor-assembly-rl`: 统一分支，已合并 CR5AF 和 Marvin
  - `serl_robot_infra/cr5af_env/` — 独立 CR5AF Gym env（不继承 FrankaEnv）
  - `serl_robot_infra/marvin_env/` — Marvin Gym env（已有）
  - `examples/experiments/motor_shaft_assembly/cr5af/` — config.py, wrapper.py
  - `examples/experiments/motor_shaft_assembly/marvin/` — config.py, wrapper.py, run_actor_thor.sh, run_learner.sh

---

## 文件变更摘要

### 新增文件
- `serl_robot_infra/cr5af_env/__init__.py`
- `serl_robot_infra/cr5af_env/envs/__init__.py`
- `serl_robot_infra/cr5af_env/envs/cr5af_env.py` — 独立 CR5AF Gym env
- `examples/experiments/motor_shaft_assembly/cr5af/config.py` — CR5AF 实验配置
- `examples/experiments/motor_shaft_assembly/cr5af/wrapper.py` — MotorShaftEnv + GripperPenaltyWrapper
- `examples/experiments/motor_shaft_assembly/marvin/__init__.py`

### 修改文件
- `serl_robot_infra/robot_servers/cr5af_server.py` — 新增 FC 力控模式、reset_joint_target 参数
- `serl_robot_infra/marvin_env/envs/marvin_env.py` — action 空间扩至 7D、增加 gripper 命令
- `examples/experiments/motor_shaft_assembly/marvin/config.py` — 重命名并适配统一结构
- `examples/experiments/motor_shaft_assembly/marvin/wrapper.py` — 重命名并适配统一结构

---

## 待解决问题

### P0 — CR5AF 控制模式适配
**问题**: CR5AF 没有原生笛卡尔阻抗控制，FC 模式与 HIL-SERL 的阻抗控制行为不同。
- HIL-SERL Franka: 连续发 delta pose → 阻抗控制器执行（弹簧-阻尼模型）
- CR5AF: FCForceMode 设 target force=0 + stiffness/damping + MovL 作为平衡点调整
- **Path A（假设）**: MovL 在 FC 模式下可以作为平衡点调整，实现类似阻抗行为
- **Path B（备选）**: FC 模式只支持单点运动，不兼容连续 delta pose 循环 → 需软件层阻抗仿真
- **验证方法**: 上电后进入 FC 模式，连续发送小步长 MovL，观察是否顺滑跟随

### P1 — 真机校准参数（全部为零，需实测）
- CR5AF: TARGET_POSE, RESET_POSE, GRASP_POSE, ABS_POSE_LIMIT_HIGH/LOW
- CR5AF: D405 相机序列号、裁剪参数 IMAGE_CROP
- Marvin: 现有参数为 placeholder，需重新测量
- 力传感器单位验证（CR5AF: SixForceValue 原始单位? Marvin: fb_joint_them/10000 单位?）

### P2 — 代码缺陷
- Marvin `vel` 可能仍为零 — 需确认 server 是否已修复，或 env 中实现有限差分
- CR5AF Jacobian 是 6x6 placeholder zeros（当前不需要，未来如需 operational-space 控制再实现）

### P3 — 后续任务
- [ ] Jetson Thor 软件环境检查（pyrealsense2, 网络, agentlace）
- [ ] 分类器训练数据量规划（成功/失败图像各需多少？）
- [ ] SpaceMouse 到货后测试示教采集
- [ ] 夹爪到货后切换 `USE_GRIPPER=True` 并验证抓取流程
- [ ] 实现剩余 5 个装配任务的环境和 wrapper

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

---

## 下一步行动（按优先级）

1. **CR5AF 上电** → 查找 IP → 测试 dashboard + RT 数据流
2. **验证 FC + MovL Path A** → 若失败，启动 Path B（软件阻抗仿真）
3. **标定工作空间** → 测量 TARGET_POSE, RESET_POSE, GRASP_POSE
4. **Jetson Thor 相机测试** → 读取 D405 序列号，验证图像流
5. **分类器数据采集** → 先手动摆位采集 50~100 张成功/失败图像
