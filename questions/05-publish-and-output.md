# 发布与输出

> 覆盖题号：Q19

### Q19（基础）：多格式发布（markdown/pdf/docx）在系统里怎么组织？为什么要做？

**标准答案（可直接说）：**
一句话：多格式发布把“研究内容”与“交付形态”解耦：先生成统一的 Markdown 布局，再按需导出 PDF/DOCX/MD，满足不同交付场景。

展开：多智能体链路中，Writer 负责生成引言/结论/目录/来源与 headers；Publisher 把章节研究结果与布局合成完整报告，并根据 `publish_formats` 写出不同格式文件（写入运行输出目录）。仓库里也有后端通用的 markdown→pdf/docx 导出能力，供 API/CLI 使用。这样设计让研究流程专注内容正确性与结构，发布模块专注格式与文件生成，可替换可扩展。

常见坑/反杀点：
- 面试官会问“为什么不让模型直接输出 PDF/DOCX”；要强调可控性与复用性。
- 导出对图片/样式敏感；当前实现能处理部分 CSS/路径，但仍有边界（优化里讲）。

**相关模块（对应实现）：**
- multi_agents/agents/publisher.py
- multi_agents/agents/utils/file_formats.py
- backend/utils.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：Publisher 先生成统一 markdown layout，再按 `publish_formats` 导出 md/pdf/docx；PDF 使用 md2pdf 并加载 CSS，DOCX 走 markdown→HTML→docx（见 `multi_agents/agents/publisher.py`、`multi_agents/agents/utils/file_formats.py`）。
- 补充：为什么 DOCX 要多一步 HTML？因为 `.docx` 本质是 Word 的一套复杂 XML 结构（段落/样式/列表/表格/图片关系等），直接从 Markdown 映射到 docx 需要自己实现大量排版与语义映射规则；引入 HTML 作为中间表示，可以先用成熟的 Markdown→HTML 解析把结构语义（标题层级、列表、引用等）稳定化，再复用现成的 HTML→docx 转换库完成常见排版映射，工程成本更低、覆盖更全、调试更直观。
- 为什么：先统一中间表示（markdown）再多格式导出，避免每种格式各写一套 Writer。
- 利弊：利是复用与一致性更好；弊是转换链路对依赖/环境敏感（PDF/DOCX 库、CSS、字体），出错时需要降级与错误提示。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 解耦的收益是什么？  
答题要点：职责清晰、便于复用、更换导出引擎更容易，减少对 LLM 输出格式依赖。

**实现：** 由同一布局导出不同格式的路径？  
答题要点：Markdown 作为中间表示；PDF 走 markdown→pdf + CSS；DOCX 走 markdown→HTML→docx。

**边界：** 导出最容易出问题的点？  
答题要点：图片路径解析、CSS 差异、字体、表格；需要更多工程化适配（可选升级）。

**取舍：** 为什么不用 HTML 作为唯一中间格式？  
答题要点：Markdown 更适合研究内容表达与 diff；HTML 渲染更强但维护成本更高，两者可组合（优化）。

**优化：** 生产级怎么做更可靠？  
答题要点：模板化渲染、字体与资源打包、导出队列化、失败重试、统一报告元数据索引（可选升级）。

---

## 附：Top12 对应条目（从原附录迁移）

（本模块无对应 Top12 条目）
