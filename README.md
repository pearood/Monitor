# 小睿伴学：基于视觉感知的在线学习专注力智能识别和生成式伴学系统

小睿伴学是一套面向在线学习场景的桌面端智能学习辅助系统。项目以视觉专注力识别为主线，结合生成式 AI 助手能力，完成“学习状态感知 + 学习内容辅助 + 多角色协同查看”的一体化闭环。

## 1. 项目定位

本项目不只是“检测学生是否分心”，而是希望把在线学习过程中的三个关键环节串起来：

1. 看见学生现在是否专注
2. 帮助教师和家长理解学生最近的学习状态
3. 让小睿直接参与学习资料整理、内容总结和复习支持

因此，小睿伴学的定位是：

**一个集专注力识别、学习内容整理和多角色协同支持于一体的智能学习助手系统。**

## 2. 当前核心功能

### 2.1 学生端

- 摄像头实时检测专注度
- 检测人脸、姿态、头部状态和手机使用情况
- 展示平均专注度、累计专注时长、分心次数
- 查看详细分析页面
- 最小化为悬浮机器人 + 小睿对话框
- 支持图片、代码、文档点击上传与拖拽上传
- 支持对当前前台网页、图片、文档做学习化处理

### 2.2 教师端

- 创建班级并生成班级码
- 查看学生班级看板
- 查看学生在线状态
- 查看班级专注度统计与学习数据

### 2.3 家长端

- 绑定自己的孩子账号
- 查看孩子最近专注状态
- 查看每日学习趋势
- 查看孩子最近向小睿提出的问题

### 2.4 小睿助手

小睿支持两种主要交互方式：

- 文本输入
- 语音输入

同时，小睿已经支持基础会话管理：

- 一键开启 `新对话`
- 历史对话分段保留
- 可按段切换查看旧对话
- 可选择彻底删除单段历史对话
- 最小化后仍可直接开启新对话、启动/关闭语音助手

当前可完成的典型任务包括：

- 查询当前专注度
- 打开详细数据页面
- 开始/停止检测
- 总结上传文档
- 解释当前图片内容
- 把当前页面整理成学习笔记
- 自动生成复习提纲
- 自动生成思维导图、流程图、框架图、结构图、架构图图片
- 自动整理 Excel 学习表格
- 自动打开指定学习网站并搜索资料
- 自动采集指定主题学习资料并保存为文档

## 3. 当前模型与算法分工

### 3.1 专注力识别主链路

专注力评分主要由视觉链路完成，核心模块包括：

- `UHMF`：主融合评分模型
- `YOLO11n`：手机目标检测
- `OpenCV LBF Facemark`：眼睛、嘴巴、人脸关键点检测
- `OpenCV solvePnP`：基于 2D-3D 人脸关键点估计头部姿态
- `PERCLOS / EAR / MAR`：闭眼比例、眼睛开合和嘴巴张开时序特征
- `RandomForest`：辅助判分与兜底修正
- `OpenCV + 规则 + EMA`：人脸检测、规则修正与平滑输出

一句话概括：

**当前项目的专注力评分是“视觉主导 + 多模块融合”的结构。**

### 3.2 生成式 AI 助手链路

当前项目中生成式 AI 的职责与专注力模型不同，主要负责“理解、总结、整理和生成”：

- `DeepSeek Chat`
  - 文本问答
  - 文件/代码解读
  - 学习笔记整理
  - 复习提纲生成
- `DeepSeek Reasoner`
  - 更深入的文本推理型问题
- `Doubao Vision`
  - 图片理解
  - 当前图片解释
  - 上传图片问答
- `Doubao Image Generation`
  - 思维导图图片生成
  - 流程图/结构图/架构图/框架图生成
- `Doubao Speech`
  - 语音播报（TTS）
  - 语音识别（ASR）

当前附件处理策略是：

- 图片理解优先：`Doubao Vision`
- 文件/代码解读优先：`DeepSeek Chat`
- 图示图片生成优先：`Doubao Image Generation 5.0 Lite`

一句话概括：

**DeepSeek 负责文本智能，Doubao 负责图片与语音能力。**

## 4. 学习化处理能力

### 4.1 自动采集学习资料

你可以直接说：

- `帮我收集高中数学函数的学习资料并保存到桌面`
- `打开 bilibili 网站并搜索某位老师内容，整理成学习资料`
- `搜索高中数学函数并整理成 Excel 表格保存到桌面`

系统会：

1. 识别主题和目标网站
2. 在征得用户同意后打开相应网站或读取网页资料
3. 提取网页正文或可读内容
4. 调用大模型整理成学习资料包
5. 输出为文档并保存

如果只需要打开网页搜索、不需要整理保存，也可以直接说：

- `打开哔哩哔哩搜索宋浩老师高等数学`
- `打开知乎搜索高中数学函数`
- `搜索二次函数学习资料`

无论是“只打开网页搜索”，还是“后台搜索并整理学习资料”，系统都会先弹窗确认是否允许小睿联网搜索、打开网页或读取网页正文；用户取消后不会访问网页。

### 4.2 当前页面/文档学习化处理

当前版本支持：

- 当前网页总结
- 当前图片解释
- 当前前台文档学习化处理
- 上传图片/代码/文档后总结
- 将当前内容或上传内容整理为分段会话中的独立学习任务

输出形式支持：

- 学习笔记
- 复习提纲
- 思维导图
- Excel 学习表格

其中思维导图会导出为 **PNG 图片**。

Excel 表格会导出为 **XLSX 文件**，适合整理知识点、公式/例题、易错点、复习建议、学习任务等结构化内容。可以直接说：

- `把当前内容整理成 Excel 表格保存到桌面`
- `把这份文档转成知识点表格`
- `把上传的图片内容整理成 Excel`

对于图片和文档学习任务，系统当前采用“先提取内容，再生成结果”的稳态链路：

1. 当前屏幕内容先由 `Doubao Vision` 进行视觉理解
2. 上传图片先由 `Doubao Vision` 提取主题、结构、文字与公式
3. 上传文档先本地提取文本内容
4. 再由 `DeepSeek` 负责学习笔记、提纲、Excel 表格等文本结构化整理

### 4.3 当前图片生成能力说明

1. **Doubao 通用图示生成**
   - 支持根据自然语言或当前页面/文档内容生成：
     - 思维导图
     - 流程图
     - 框架图
     - 结构图
     - 架构图

### 4.4 保存方式

支持两种保存模式：

- 保存到桌面
- 自定义保存路径

例如：

- `把当前页面整理成学习笔记并保存到桌面`
- `把这个文档生成复习提纲，我自己选保存位置`

## 5. 角色与演示账号

当前默认演示账号：

- 学生：`student1 / 123456`
- 教师：`teacher1 / 123456`
- 家长：`parent1 / 123456`

## 6. 运行方式

### 6.1 安装依赖

```bash
cd "/Users/m/Desktop/focus/xiaorui-study-companion"
pip install -r requirements.txt
pip install -r requirements-voice.txt
```

其中 `opencv-contrib-python` 用于提供 LBF 人脸关键点检测能力。

如果是 macOS 且 `PyAudio` 安装失败，可先安装：

```bash
brew install portaudio
pip install PyAudio==0.2.14
```

### 6.2 启动程序

```bash
cd "/Users/m/Desktop/focus/xiaorui-study-companion"
python3 main.py
```

## 7. 生成式模型配置

### 7.1 DeepSeek

复制模板并填写：

```bash
cp "/Users/m/Desktop/focus/xiaorui-study-companion/config/deepseek.env.example" \
   "/Users/m/Desktop/focus/xiaorui-study-companion/config/deepseek.env"
```

### 7.2 Doubao Vision / Speech

复制模板并填写：

```bash
cp "/Users/m/Desktop/focus/xiaorui-study-companion/config/doubao.env.example" \
   "/Users/m/Desktop/focus/xiaorui-study-companion/config/doubao.env"
```

当前 Doubao 配置分成三类：

- `DOUBAO_API_KEY`
  - 用于 Doubao Vision 与 Doubao Image Generation
- `DOUBAO_TTS_VERSION=2` + `DOUBAO_SPEECH_APPID` + `DOUBAO_TTS_API_KEY`
  - 用于豆包语音合成 2.0 播报（TTS），当前推荐方式
- `DOUBAO_SPEECH_TOKEN` + `DOUBAO_TTS_CLUSTER`
  - 用于旧版 TTS 1.0 回退
- `DOUBAO_ASR_API_KEY`
  - 用于语音识别（ASR）

当前语音链路的策略是：

- 语音识别可优先接入 Doubao ASR
- 语音播报可优先接入 Doubao TTS 2.0，接口为 `/api/v3/tts/unidirectional`
- 如果 Doubao 语音服务不可用，系统会自动回退到本地语音链路，保证小睿仍可继续工作

TTS 2.0 推荐配置示例：

```env
DOUBAO_TTS_VERSION=2
DOUBAO_SPEECH_APPID=支持语音合成2.0的应用APPID
DOUBAO_TTS_API_KEY=语音合成2.0 API Key
DOUBAO_TTS_RESOURCE_ID=seed-tts-2.0
DOUBAO_TTS_VOICE_TYPE=zh_female_xiaohe_uranus_bigtts
DOUBAO_TTS_ENCODING=mp3
DOUBAO_TTS_RATE=24000
```

当前界面支持切换小睿播报声音，内置音色包括：

- 小何 2.0：`zh_female_xiaohe_uranus_bigtts`
- Vivi 2.0：`zh_female_vv_uranus_bigtts`
- 云舟 2.0：`zh_male_m191_uranus_bigtts`
- 小天 2.0：`zh_male_taocheng_uranus_bigtts`

## 8. 云端能力

项目支持远程后端同步，当前云端能力包括：

- 登录注册
- 学生专注记录同步
- 教师班级与学生数据同步
- 家长绑定孩子与家长看板
- 小睿提问记录同步
- 小睿历史会话分段同步

### 8.1 数据存储说明

当前项目默认开启云端同步，登录注册、专注记录、班级信息、家长看板、小睿提问记录和历史对话会优先写入远程后端。

项目仍保留本地 SQLite 模式，用于离线运行、调试或关闭远程接口时使用；因此代码中仍能看到 `runtime/users.db` 和本地数据库逻辑，但在默认云端模式下，核心业务数据以服务器为主。

此外，以下内容属于本地文件，不是云端数据库数据：

- DeepSeek / Doubao 的本地配置文件
- 用户主动导出的学习笔记、Word 文档、思维导图或图片
- 本地模型资源、语音资源和缓存文件

## 9. 项目目录说明

```text
xiaorui-study-companion/
├── app/
│   ├── core/        # 检测、数据库、语音、DeepSeek/Doubao、学习助手等核心逻辑
│   ├── ui/          # 登录页、学生端、教师端、家长端、分析页
│   └── utils/       # 数据增强等辅助工具
├── assets/
│   └── branding/    # 应用 Logo、图标等品牌素材
├── archive/         # 精简保留的 UHMF 模型文件与答辩图表资料
├── config/          # DeepSeek / Doubao 本地配置模板
├── data/            # 数据集
├── models/          # 必要本地模型与语音资源
├── deploy/          # Nginx、PyInstaller 等部署/打包配置
├── runtime/         # 本地运行数据，例如离线 SQLite 数据库
├── server/          # 云端 API
├── website/         # 下载官网页面资源
├── scripts/         # 打包、发布、部署脚本
├── requirements.txt
├── requirements-voice.txt
└── main.py
```

## 10. 当前边界说明

为了让项目行为更真实稳定，这里也明确写出当前边界：

- 当前页面、图片、文档总结当前采用“用户授权截图 + 视觉模型理解”的方式；它理解的是当前屏幕可见内容，长文档需要滚动后再次处理
- 授权截图会作为临时文件传入视觉理解链路，分析完成、失败或取消后自动清理；用户主动上传的文件不会被自动删除
- 语音唤醒和打断已做近音词、短句和播报中断优化，但实际效果仍会受麦克风、环境噪声和系统录音权限影响
- 图片理解目前优先由 `Doubao Vision` 承担
- 文件/代码解读目前优先由 `DeepSeek` 承担
- 通用图示图片生成目前优先由 `Doubao Image Generation (5.0 Lite)` 承担，复杂图示的稳定性仍取决于提示词质量与模型服务可用性
- 最小化对话框当前以快速使用为主，历史会话管理的主视图仍以学生端主界面为准

## 11. 项目价值

小睿伴学的核心价值不只是“看见学生专不专注”，还在于：

- 让教师和家长能从不同视角理解学生状态
- 让学生能直接借助小睿完成资料整理与学习辅助
- 把“专注力识别”与“学习内容支持”整合进同一套桌面系统

**小睿伴学是一套把学习状态感知、内容理解整理和多角色协同支持结合起来的智能学习辅助平台。**
