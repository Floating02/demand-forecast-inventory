# 贡献指南

感谢您对本项目的关注和支持！我们欢迎各种形式的贡献，包括但不限于：

## 如何贡献

### 1. 报告问题

如果您发现了bug或者有新功能建议，请在 GitHub Issues 中提交：

- **Bug 报告**：请详细描述问题，包括复现步骤、预期行为和实际行为
- **功能建议**：请说明新功能的用途和实现思路

### 2. 提交代码

1. **Fork 仓库**：在 GitHub 上 fork 本项目
2. **克隆仓库**：
   ```bash
   git clone https://github.com/your-username/demand-forecast-inventory.git
   cd demand-forecast-inventory
   ```
3. **创建分支**：
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **进行修改**：实现您的功能或修复
5. **运行测试**：确保您的修改不会破坏现有功能
6. **提交代码**：
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
7. **推送分支**：
   ```bash
   git push origin feature/your-feature-name
   ```
8. **创建 Pull Request**：在 GitHub 上提交 PR

### 3. 代码规范

- 遵循 Python PEP 8 代码规范
- 保持代码简洁明了，添加必要的注释
- 确保代码能够通过现有的测试
- 为新功能添加相应的测试

### 4. 文档贡献

- 改进 README.md 或其他文档
- 添加使用示例
- 补充 API 文档

## 开发环境设置

### 安装依赖

```bash
pip install pandas numpy scipy scikit-learn openpyxl
```

### 运行测试

```bash
# 运行数据分析
python analyze_data.py

# 运行需求分析
python demand_analysis.py

# 运行常规预测
python demand_forecast.py

# 运行新品预测
python forecast_new_products.py

# 运行促销预测
python forecast_promo.py
```

## 行为准则

请参考 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) 了解我们的行为准则。

## 联系方式

如有任何问题，您可以：
- 在 GitHub Issues 中提问
- 联系项目维护者

再次感谢您的贡献！