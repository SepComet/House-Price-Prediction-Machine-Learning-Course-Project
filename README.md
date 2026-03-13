# 房价预测机器学习项目

这个仓库按 [ProjectDesign.md](./ProjectDesign.md) 初始化，用于完成 California Housing 数据集上的探索分析、回归建模、评估和可视化。
当前已经包含一个可扩展的主流程骨架：读取数据集、划分训练测试集、按统一接口执行模型、汇总评估结果并输出报表。

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
│       ├── __main__.py
│       └── config.py
├── tests/
│   ├── test_config.py
│   └── test_pipeline.py
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

## 运行方式

推荐使用模块方式运行主流程：

```powershell
python -m house_price_ml
```

如果你没有执行 `pip install -e ".[dev]"`，也可以直接运行入口文件：

```powershell
python src\house_price_ml\__main__.py
```

主流程当前会执行这些步骤：

1. 读取 California Housing 数据集
2. 保存原始数据到 `data/raw/california_housing.csv`
3. 生成数据集摘要
4. 划分训练集和测试集
5. 依次调用各个模型接口
6. 输出汇总结果到 `reports/` 和 `data/processed/`

当前默认注册了 3 个回归模型：

- `LinearRegression`
- `DecisionTreeRegressor`
- `RandomForestRegressor`

- `LinearRegression`：使用标准化 + 线性回归
- `DecisionTreeRegressor`：单棵决策树回归
- `RandomForestRegressor`：随机森林回归

这些模型已经接入训练与评估流程，运行后会输出真实的 `MAE`、`MSE`、`RMSE` 和 `R²`。

## 输出文件

主流程运行成功后会生成：

- `data/raw/california_housing.csv`：原始数据集导出
- `data/processed/dataset_summary.csv`：数据集摘要
- `reports/model_results.csv`：模型结果表
- `reports/training_summary.json`：完整汇总 JSON

## 快速验证

安装依赖后运行：

```powershell
python scripts/check_env.py
python -m pytest
```

`check_env.py` 会验证关键库是否可导入，并尝试加载 California Housing 数据集。
脚本会把运行时缓存放到项目内的 `.cache/` 目录，避免依赖用户主目录权限。

首次运行 `check_env.py` 时，如果本地还没有 California Housing 数据集，需要联网下载一次。

如果你要验证主流程入口，可以再执行：

```powershell
python -m house_price_ml
```

## 注意事项

- 建议优先使用 `python -m house_price_ml`，这是标准包运行方式。
- `python src\house_price_ml\__main__.py` 现在也兼容，但更适合作为备用方式。
- California Housing 数据集第一次获取时需要联网；如果网络受限且本地没有缓存，程序会直接提示数据加载失败。
- 数据集缓存目录为 `.cache/scikit_learn_data/`。
- 当前测试可以通过，但如果工作区对 `.pytest_cache` 写入受限，`pytest` 可能会输出缓存警告；这不影响测试结果。
- 当前模型层已经提供统一接口，后续如果要继续扩展算法，只需要新增实现 `fit_predict()` 的模型类并注册到默认模型列表。

## 清理缓存

下面这些目录都是运行过程中可能生成的缓存或临时文件，可以按需删除：

- `.cache/`
- `.tmp/`
- `.pip-cache/`
- `.pytest_cache/`
- `pytest-cache-files-*`

## 下一步建议

1. 在 `src/house_price_ml/models.py` 中继续扩展更多模型，例如 Ridge、Lasso、SVR。
2. 在 `notebooks/` 中补充数据探索和可视化分析。
3. 继续扩展测试，覆盖数据读取、报表导出和完整流水线。
