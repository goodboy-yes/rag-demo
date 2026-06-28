# raglite-demo

一个轻量 RAG 向量检索学习项目。第一版只实现检索核心，不调用大语言模型：

1. 读取 `md/txt/jsonl` 文档
2. 切分为带 overlap 的 chunk
3. 使用本地 embedding 模型生成向量
4. 保存本地索引
5. 根据查询返回 top-k 相关片段

## 为什么不需要 Llama

本项目使用的是 embedding 模型，不是聊天模型。`BAAI/bge-small-zh-v1.5` 的职责是把文本转换成 512 维向量；检索时把问题也转换成向量，再用余弦相似度找到最相关的文档片段。

WSL 里安装的 `llama-cpp-python` 适合后续做“检索结果 + 生成答案”的完整 RAG。当前版本专注向量检索，所以不接 Llama。

## 模型选择

默认模型是 `BAAI/bge-small-zh-v1.5`，通过 FastEmbed/ONNX Runtime 在本地运行。

选择它的原因：

- 面向中文语义检索，比英文 small embedding 模型更适合中文资料。
- v1.5 相比旧版改善了相似度分布，也更适合无 instruction 的检索场景。
- 体积小，约 24M 参数、约 95.8MB，输出 512 维向量。
- 对电脑配置要求低，小型 demo 用 CPU 即可运行；4 核 CPU、4GB 内存通常够用。

索引大小可以粗略估算为：

```text
chunk_count * 512 * 4 bytes
```

例如 10,000 个 chunk 的原始向量矩阵约 20MB。

## WSL 环境

Windows 路径：

```powershell
D:\code\rag\demo1
```

WSL 路径：

```bash
/mnt/d/code/rag/demo1
```

创建隔离环境。这里使用 WSL 中已经缓存过的 Python 3.10.20：

```bash
wsl -d Ubuntu-22.04
cd /mnt/d/code/rag/demo1
/home/aibox/miniconda3/bin/conda create -n raglite python=3.10.20 --offline -y
/home/aibox/miniconda3/bin/conda activate raglite
```

如果当前 shell 没有初始化 conda，可以改用：

```bash
source /home/aibox/miniconda3/etc/profile.d/conda.sh
conda activate raglite
```

安装当前项目及依赖：

```bash
pip install -e ".[dev]"
```

这条命令可以拆成几个部分理解：

- `pip install`：让 pip 安装指定目标。目标可以是第三方包、依赖文件，也可以是当前项目。
- `-e`：`--editable` 的缩写，表示可编辑安装。
- `.`：当前目录，也就是 `/mnt/d/code/rag/demo1` 这个项目。
- `[dev]`：安装 `pyproject.toml` 里定义的 `dev` 可选依赖组。
- `"..."`：引号用于避免 shell 把 `[]` 误解成特殊字符，在 PowerShell、bash 等环境里更稳。

所以：

```bash
pip install -e ".[dev]"
```

完整意思是：以可编辑模式安装当前目录这个项目，同时安装它的 `dev` 开发/测试依赖。

它会安装三类东西：

- 当前项目本身：`raglite-demo`
- 运行依赖：`fastembed`、`httpx[socks]`、`numpy`、`rich`、`typer`
- 开发测试依赖：`pytest`

`pip install` 不只是安装依赖，它安装什么取决于参数：

```bash
pip install numpy              # 安装第三方包
pip install -r requirements.txt # 按依赖文件安装
pip install .                  # 安装当前目录这个项目
pip install -e ".[dev]"         # 可编辑安装当前项目，并安装 dev 可选依赖
```

这条命令会把项目安装到当前激活的 Python 环境里。这里指的是 WSL 的 conda 环境：

```bash
/home/aibox/miniconda3/envs/raglite
```

它不会安装到 Windows Python、WSL 系统 Python，也不会安装到其他 conda 环境，例如 `base`、`llm`、`fine-tuning`。

激活 `raglite` 环境后，在 WSL 的任意目录都可以导入项目代码：

```python
from raglite.store import search_index
```

也可以在任意目录运行命令行工具：

```bash
raglite --help
```

这是因为命令行入口被安装到了当前环境的 `bin` 目录：

```bash
/home/aibox/miniconda3/envs/raglite/bin/raglite
```

`conda activate raglite` 会把这个目录加入 `PATH`。如果没有激活环境，也可以用完整路径运行：

```bash
/home/aibox/miniconda3/envs/raglite/bin/raglite --help
```

WSL 和 Windows 是两套环境。当前项目安装在 WSL 的 conda 环境里，所以 Windows PowerShell 默认不能直接运行 `raglite --help`，Windows Python 也不能直接 `import raglite`。如果确实想在 Windows Python 中使用，需要在 Windows 对应的 Python 环境里重新安装。

因为这里使用了 `-e` 可编辑安装，环境里不是复制一份固定代码，而是指向当前项目目录：

```bash
/mnt/d/code/rag/demo1
```

所以修改 `raglite/cli.py`、`raglite/store.py` 等源码后，不需要重新安装就能生效。但如果移动或删除这个项目目录，当前环境里的可编辑安装也会失效。

安装后可以这样验证项目本身已经进入当前环境：

```bash
pip show raglite-demo
python -c "import raglite; print(raglite.__file__)"
raglite --help
```

如果 `pip show` 能看到 `raglite-demo`，`python -c` 能打印 `raglite` 的文件路径，`raglite --help` 能显示命令帮助，就说明项目本身、Python 导入和命令行入口都已经安装成功。

## pyproject.toml 说明

`pyproject.toml` 是 Python 项目的标准配置文件。它告诉 `pip` 怎么构建项目、项目叫什么、支持哪些 Python 版本、需要安装哪些依赖、命令行入口在哪里，以及测试工具如何运行。

判断一个目录是不是可以被 `pip install .` 安装，通常看项目根目录有没有这些标准配置文件：

```text
pyproject.toml
setup.py
setup.cfg
```

现代 Python 项目推荐使用 `pyproject.toml`。如果只有普通 `.py` 文件，而没有这些配置，代码仍然可以通过 `python some_script.py` 运行，但 pip 通常不知道项目名、版本、依赖、要安装哪些包、是否要生成命令行工具，因此不能把它当成标准项目安装。

本项目的 `pyproject.toml` 分成几个配置块：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

这部分说明项目如何被构建。

- `requires`：构建项目本身需要先安装的工具，这里使用 `setuptools`。
- `build-backend`：构建后端，这里表示让 `setuptools` 负责打包和安装。

`setuptools` 没有放到 `[project].dependencies`，因为它不是项目运行时依赖，而是安装/打包项目时使用的构建工具。`pip install .` 时，pip 会先根据 `[build-system].requires` 准备构建环境并安装 `setuptools>=68`，然后再用它构建当前项目。

可以把两类依赖分开理解：

```text
[build-system].requires
  安装/打包这个项目时需要的工具
  例如 setuptools

[project].dependencies
  用户运行这个项目时需要的库
  例如 fastembed、numpy、typer
```

`build-backend = "setuptools.build_meta"` 的意思是：当 pip 安装/构建项目时，使用 `setuptools` 提供的 `build_meta` 构建后端。`build_meta` 不是本项目里的文件，而是 `setuptools` 自带的模块，它实现了 Python 标准构建接口，负责读取项目元信息、找到包、生成 wheel、处理可编辑安装和命令行入口。

这来自 Python 的 PEP 517 / PEP 518 构建标准。pip 本身不规定每个项目必须怎么构建，而是读取 `build-backend`，再调用对应后端。其他项目也可能使用别的构建后端，例如 `hatchling.build` 或 `poetry.core.masonry.api`。

```toml
[project]
name = "raglite-demo"
version = "0.1.0"
description = "A lightweight RAG vector retrieval CLI demo."
readme = "README.md"
requires-python = ">=3.10,<3.14"
dependencies = [...]
```

这部分是项目元信息和运行依赖。

- `name`：包名，`pip show raglite-demo` 会看到这个名字。
- `version`：项目版本号。
- `description`：一句话描述。
- `readme`：项目说明文档路径。
- `requires-python`：支持的 Python 版本范围；本项目用 WSL 中已缓存的 Python 3.10.20。
- `dependencies`：运行项目必须安装的依赖。

运行依赖含义：

- `fastembed`：加载 embedding 模型并把文本转成向量。
- `httpx[socks]`：让 Hugging Face 下载模型时支持 SOCKS 代理。
- `numpy`：保存向量矩阵、归一化向量、计算相似度。
- `rich`：让 CLI 输出表格更清楚。
- `typer`：构建 `raglite ingest/search/inspect` 命令行。

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8",
]
```

这部分定义可选依赖。`dev` 表示开发/测试依赖，所以执行：

```bash
pip install -e ".[dev]"
```

会安装当前项目本身、运行依赖，以及 `dev` 里的 `pytest`。

如果只执行：

```bash
pip install .
```

就会安装当前项目本身和运行依赖，但不会安装 `dev` 里的 `pytest`。

```toml
[project.scripts]
raglite = "raglite.cli:app"
```

这部分定义命令行入口。安装后，系统会生成一个 `raglite` 命令，运行时会加载 `raglite/cli.py` 里的 `app`。

它负责的是终端命令：

```bash
raglite --help
raglite search "什么是向量检索"
```

```toml
[tool.setuptools.packages.find]
include = ["raglite*"]
```

这部分告诉 `setuptools` 哪些 Python 包需要被安装。这里会安装 `raglite` 包及其子包。

它负责的是 Python 导入：

```python
from raglite.store import search_index
```

所以两者区别是：

```text
[tool.setuptools.packages.find]
include = ["raglite*"]
  决定哪些 Python 包会被安装进环境
  影响 import raglite、from raglite.store import search_index

[project.scripts]
raglite = "raglite.cli:app"
  决定安装后生成什么命令行命令
  影响 raglite --help、raglite search ...
```

这两部分也有关联：`[project.scripts]` 里写的是 `raglite.cli:app`，所以运行命令时仍然需要 `raglite` 包能够被安装和导入。否则命令即使生成了，执行时也会因为找不到 `raglite.cli` 而失败。

在 Python 里，包、模块、子包可以这样理解：

```text
raglite/
  __init__.py
  store.py
  documents.py
```

- `raglite/` 目录里有 `__init__.py`，所以它是一个 Python 包。
- `store.py`、`documents.py` 是 `raglite` 包里的模块。
- 如果以后新增 `raglite/vector/__init__.py`，那么 `raglite.vector` 就是 `raglite` 的子包。

导入模块和子包用的是同一套 import 语法，但 `from ...` 后面必须接 `import ...`。下面这些写法是有效的：

```python
import raglite.store

from raglite.store import search_index
from raglite.documents import load_documents
```

如果以后有这样的子包结构：

```text
raglite/
  vector/
    __init__.py
    index.py
```

就可以这样导入：

```python
import raglite.vector

from raglite.vector import index
from raglite.vector.index import VectorIndex
```

下面这种写法是不完整的：

```python
from raglite.store
from raglite.vector
```

需要补上 `import` 后面的对象，例如：

```python
from raglite.store import search_index
from raglite.vector import index
```

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

这部分是 pytest 配置。执行 `pytest` 时，默认只从 `tests/` 目录收集测试。

## WSL 代理

你已经在 Clash Verge 中开启了局域网连接。当前验证结果：

- Windows 代理进程：`verge-mihomo.exe`
- WSL 网关：`172.28.144.1`
- `http://172.28.144.1:7890` 可以访问 Hugging Face
- `socks5h://172.28.144.1:7890` 可以访问 Hugging Face
- 不走代理访问 Hugging Face 会超时

临时设置代理：

```bash
export http_proxy=http://172.28.144.1:7890
export https_proxy=http://172.28.144.1:7890
export all_proxy=socks5h://172.28.144.1:7890
```

WSL 重启后网关可能变化，可以动态获取：

```bash
export WIN_GATEWAY=$(ip route | awk '/default/ {print $3; exit}')
export http_proxy=http://$WIN_GATEWAY:7890
export https_proxy=http://$WIN_GATEWAY:7890
export all_proxy=socks5h://$WIN_GATEWAY:7890
```

验证代理：

```bash
curl -I -L -x http://$WIN_GATEWAY:7890 https://huggingface.co
```

## 使用

构建索引：

```bash
raglite ingest examples/docs --index .raglite --chunk-size 500 --overlap 80
```

查看索引：

```bash
raglite inspect --index .raglite
```

搜索：

```bash
raglite search "什么是向量检索" --top-k 5 --index .raglite
```

也可以直接用模块方式运行：

```bash
python -m raglite search "什么是向量检索" --top-k 5 --index .raglite
```

## 输入格式

支持三类文件：

- `.md`
- `.txt`
- `.jsonl`

JSONL 每一行必须是 JSON object，优先读取 `text` 字段，其次读取 `content` 字段。其他字段会保存为 metadata。

示例：

```jsonl
{"text":"向量检索会把文本映射到语义空间。","title":"vector-search"}
{"content":"chunk overlap 可以缓解上下文被切断的问题。","title":"chunking"}
```

## 索引文件

默认保存到 `.raglite/`：

- `index.npy`：float32 向量矩阵
- `chunks.jsonl`：chunk 文本和来源信息
- `manifest.json`：模型名、维度、chunk 参数、创建时间

当前版本每次 `ingest` 都会重建索引，不做增量更新。

## 测试

```bash
pytest
```

测试使用 fake embedder，不会下载真实模型。
