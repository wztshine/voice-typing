# 语音打字 (voice-typing)

按住一个键说话，松开自动把文字打到光标处。本地识别，不上传任何数据。

- **快**：RTX 2060 上说 5 秒话约 1 秒出字（FireRedASR2-AED 1.1B，fp16）
- **剪贴板不毁**：粘贴后你之前复制的图片/文件/富文本完整还原，识别文本也不进 Win+V 历史
- **输出润色**：数字口语转换（"四千五"→4500、"一万二"→12000）、去"嗯啊"、短句自动去句号（问号保留）
- **热词**：专有名词解码偏置 + 大小写校正

## 快速开始（新电脑）

要求：**Windows + NVIDIA 显卡（≥4GB 显存）+ Python 3.10~3.12**，联网下载约 6GB。

```bat
git clone https://github.com/wztshine/voice-typing.git
cd voice-typing
双击「安装环境.bat」      :: venv + CUDA PyTorch + 依赖 + 模型下载(支持断点续传)
双击「启动语音输入.bat」   :: 首次加载约 20 秒, 浮窗显示「就绪」即可用
```

> PyTorch 必须装 CUDA 版（脚本已处理）；各代 N 卡对照表、GTX 10 系陷阱、FP8 说明见
> [docs/进阶指南.md](docs/进阶指南.md#显卡兼容性)。

## 使用

| 按键 | 作用 |
|---|---|
| **右 Ctrl 按住说话，松开** | 识别并输入（全新安装默认 F7，双击 `设置热键.bat` 改） |
| **F10** | 按一下开始、再按一下结束（长段口述用） |

- 录音/识别/完成状态看屏幕底部浮窗，开始/结束有提示音
- 单次最长 120 秒；长段口述的停顿会被自动断句合并
- 识别走「剪贴板 + Ctrl+V」，短句自动去句号（问号/叹号保留）

## 常用配置（config.json，改完双击「重启语音输入.bat」）

```json
{
  "hotkey": "ctrl_r",
  "toggle_key": "f10",
  "hotword_bonus": 2.0,
  "trailing_punct": "auto",
  "number_norm": true,
  "filler_removal": true,
  "beam_size": 3,
  "max_sec": 120
}
```

| 想调什么 | 改哪个 | 说明 |
|---|---|---|
| 说话键 | `hotkey` | 右 Ctrl / F1~F24 / scroll_lock…，可视化选择用 `设置热键.bat` |
| 长口述开关 | `toggle_key` | F10 按一下开始再按结束；`none` 关闭 |
| 热词强度 | `hotword_bonus` | 2.0 默认；4~6 更强势但易误伤 |
| 短句去句号 | `trailing_punct` | `auto`/`strip`/`keep`；阈值 `trailing_punct_max_len`(15)；永不删的符号 `trailing_punct_keep` |
| 数字口语转换 | `number_norm` | 四千五→4500、十万八→108000；"一点/两点"不误转 |
| 去"嗯啊" | `filler_removal` | 口语冗余删除 |
| 速度⇄精度 | `beam_size` | `1` 快约一倍，`3` 更准 |
| 最长录音 | `max_sec` | 秒 |
| 浮窗 | `show_ui` | 关掉省一点 CPU |

完整 30+ 项配置表、热词/替换表格式、断句调优、性能实测 → [docs/进阶指南.md](docs/进阶指南.md)

## 常见问题

- **没反应/听不到**：`查看状态.bat` 看日志；麦克风设备号用 `--list-devices` 查
- **慢**：确认是 CUDA 版 torch（`python -c "import torch; print(torch.version.cuda)"`）；`beam_size` 改 1
- **GTX 10 系老卡**：必须 `"use_half": false, "fast_load": false`（Pascal 的 FP16 慢几十倍）
- **程序无声消失**：查 `logs/voice_typing.log` 里 `程序异常退出:` 段
- 更多排查 → [docs/进阶指南.md](docs/进阶指南.md#排查识别结果不对)

## 目录结构

```
voice_typing.py        主程序
fireredasr2s/          识别库源码(内置, 含内存直传补丁)
hotword.py             热词(解码偏置 + 大小写校正)
postprocess.py         输出后处理(数字/标点/口语规则)
dl.py                  模型下载(ModelScope, ~6GB)
make_fp16.py           fp32→fp16 权重转换
安装环境.bat           一键装环境 + 模型
启动/停止/重启.bat      日常操作
设置热键.bat           可视化选说话键
hotwords.txt           热词表(个人文件, 模板: hotwords.example.txt)
replacements.txt       自定义替换表
config.json            用户配置(个人文件)
```
