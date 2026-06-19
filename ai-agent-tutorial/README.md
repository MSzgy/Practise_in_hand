# AI Agent 架构与框架教程

一套系统学习 AI Agent 架构和主流框架的 Jupyter Notebook 教程，**基于 ModelScope + Qwen 开源模型，可直接运行**。

## 教程结构

```
ai-agent-tutorial/
├── 01_core_concepts/          # 核心概念
│   ├── 00_what_is_agent.ipynb      # Agent 基础
│   ├── 01_react_and_cot.ipynb      # ReAct 与 CoT
│   ├── 02_tool_use_and_function_calling.ipynb  # 工具使用
│   ├── 03_memory_systems.ipynb     # 记忆系统
│   └── 04_planning_and_reflection.ipynb        # 规划与反思
├── 02_frameworks/             # 主流框架
│   ├── 00_langchain_basics.ipynb   # LangChain 基础
│   ├── 01_langgraph_workflows.ipynb # LangGraph 工作流
│   ├── 02_autogen_multi_agent.ipynb # AutoGen 多 Agent
│   └── 03_crewai_collaboration.ipynb # CrewAI 协作
├── 03_advanced/               # 高级主题
│   ├── 00_multi_agent_systems.ipynb  # 多 Agent 系统
│   ├── 01_rag_agent.ipynb           # RAG Agent
│   └── 02_tool_use_advanced.ipynb   # 高级工具使用
├── 04_projects/               # 实战项目
│   ├── 00_knowledge_base_qa.ipynb    # 知识库问答
│   └── 01_research_assistant.ipynb   # 研究助手
├── requirements.txt
└── README.md
```

## 学习目标

1. **理解 Agent 核心架构**：ReAct、CoT、Reflection、Tool Use、Planning
2. **掌握主流框架**：LangChain、LangGraph、AutoGen、CrewAI
3. **实现高级功能**：RAG、多 Agent 协作、工具链
4. **完成实战项目**：知识库问答、研究助手

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖包括：`jupyter`、`numpy`、`torch`、`transformers`、`modelscope`、`accelerate`

### 2. 模型说明

本教程默认使用 **Qwen2.5-7B-Instruct**（通过 ModelScope 下载），首次运行会自动下载模型。

| 模型 | 显存需求 | 适用场景 |
|------|---------|---------|
| `Qwen/Qwen2.5-7B-Instruct`（默认） | ~14-16GB（FP16） | 推荐，综合性能最佳 |
| `Qwen/Qwen2.5-3B-Instruct`（低显存备选） | ~6-8GB（FP16） | 消费级 GPU |
| `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | ~4-6GB | 低显存量化版 |

如需更换模型，修改每个 Notebook 开头的 `MODEL_NAME` 变量即可。

### 3. 启动 Jupyter

```bash
cd ai-agent-tutorial
jupyter notebook
```

### 4. 按顺序学习

每个 Notebook 开头会自动加载 Qwen 模型，后续 cell 直接调用 `llm.chat()` 即可获得真实 LLM 输出。

## 学习路径

### 第一阶段：核心概念（1-2 周）
- Agent 基础与分类
- ReAct 推理行动循环
- Chain-of-Thought 思维链
- 工具使用与 Function Calling
- 记忆系统设计
- 规划与反思机制

### 第二阶段：主流框架（2-3 周）
- LangChain 组件与链式调用
- LangGraph 状态图工作流
- AutoGen 对话式多 Agent
- CrewAI 角色驱动协作

### 第三阶段：高级主题（1-2 周）
- 多 Agent 系统架构
- RAG 检索增强生成
- 高级工具使用模式

### 第四阶段：实战项目（1-2 周）
- 知识库问答系统
- 智能研究助手

## 技术特点

- **真实 LLM 调用**：通过 ModelScope 加载 Qwen2.5 开源模型，所有代码可直接运行
- **Mock 备选方案**：每个 Notebook 保留 Mock 实现作为注释，无 GPU 时可切换
- **GPU/CPU 自适应**：自动检测设备，支持 CUDA、Apple Silicon MPS、CPU
- **渐进式学习**：从基础概念到复杂系统逐步深入
- **中文内容**：全中文教程，适合中文学习者

## 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 无（CPU 可运行，速度较慢） | NVIDIA GPU 16GB+ 显存 |
| 内存 | 8GB | 16GB+ |
| 磁盘 | 20GB（模型缓存） | 50GB+ |

## Function Calling 说明

Qwen2.5-7B-Instruct 原生支持 Function Calling，但需要通过 vLLM 或 Qwen-Agent 框架才能获得开箱即用的 FC 支持。本教程中涉及 Function Calling 的部分保留了基于关键词匹配的 Mock 实现，并提供了升级到真实 FC 的说明。

如需体验完整 Function Calling，可参考：
```bash
# 使用 vLLM 部署（支持原生 FC）
export VLLM_USE_MODELSCOPE=True
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
```

## 参考资源

- [ModelScope 平台](https://modelscope.cn/)
- [Qwen2.5 模型](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)
- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [AutoGen 文档](https://microsoft.github.io/autogen/)
- [CrewAI 文档](https://docs.crewai.com/)

## 许可证

MIT License
