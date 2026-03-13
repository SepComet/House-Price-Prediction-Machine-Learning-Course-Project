# 房价预测 Python 开发环境

这个仓库按 [ProjectDesign.md](./ProjectDesign.md) 初始化，用于完成 California Housing 数据集上的探索分析、回归建模、评估和可视化。

## 目录结构

```text
.
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
├── reports/
│   └── figures/
├── scripts/
│   └── check_env.py
├── src/
│   └── house_price_ml/
│       ├── __init__.py
│       └── config.py
├── tests/
│   └── test_config.py
├── ProjectDesign.md
├── pyproject.toml
└── requirements.txt
```

## 推荐环境

- Python `3.10+`
- 开发工具：VS Code / PyCharm / Jupyter Notebook

## 安装依赖

当前工作区已经验证可在 Windows PowerShell 中直接使用 `python` 安装依赖。

如果你使用系统 Python：

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

如果你想使用虚拟环境，可以先创建并激活：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

如果你只想安装运行实验所需的最小依赖，也可以执行：

```powershell
python -m pip install -r requirements.txt
```

## 快速验证

安装依赖后运行：

```powershell
python scripts/check_env.py
python -m pytest
```

`check_env.py` 会验证关键库是否可导入，并尝试加载 California Housing 数据集。
脚本会把运行时缓存放到项目内的 `.cache/` 目录，避免依赖用户主目录权限。

首次运行 `check_env.py` 时，如果本地还没有 California Housing 数据集，需要联网下载一次。

## 清理缓存

下面这些目录都是运行过程中可能生成的缓存或临时文件，可以按需删除：

- `.cache/`
- `.tmp/`
- `.pip-cache/`
- `pytest-cache-files-*`

## 下一步建议

1. 在 `notebooks/` 中建立数据探索 notebook。
2. 在 `src/house_price_ml/` 中逐步补充数据处理、训练和评估模块。
3. 用 `tests/` 保存基础单元测试，先覆盖配置和数据读取逻辑。
