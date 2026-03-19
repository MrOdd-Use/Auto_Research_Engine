# Auto_Research_Engine Evaluations

This directory contains evaluation tools and frameworks for assessing the performance of Auto_Research_Engine across different research tasks.

## Simple Evaluations (`simple_evals/`)

The `simple_evals` directory contains a straightforward evaluation framework adapted from [OpenAI's simple-evals system](https://github.com/openai/simple-evals), specifically designed to measure short-form factuality in large language models. Our implementation is based on OpenAI's [SimpleQA evaluation methodology](https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py), following their zero-shot, chain-of-thought approach while adapting it for Auto_Research_Engine's specific use case.

### Components

- `simpleqa_eval.py`: Core evaluation logic for grading research responses
- `run_eval.py`: Script to execute evaluations against Auto_Research_Engine
- `requirements.txt`: Dependencies required for running evaluations

### Test Dataset

The `problems/` directory contains the evaluation dataset:

- `Simple QA Test Set.csv`: A comprehensive collection of factual questions and their correct answers, mirrored from OpenAI's original test set. This dataset serves as the ground truth for evaluating Auto_Research_Engine's ability to find and report accurate information. The file is maintained locally to ensure consistent evaluation benchmarks and prevent any potential upstream changes from affecting our testing methodology.

### Evaluation Logs

The `logs/` directory contains detailed evaluation run histories that are preserved in version control:

- Format: `SimpleQA Eval {num_problems} Problems {date}.txt`
- Example: `SimpleQA Eval 100 Problems 2-22-25.txt`

These logs provide historical performance data and are crucial for:
- Tracking performance improvements over time
- Debugging evaluation issues
- Comparing results across different versions
- Maintaining transparency in our evaluation process

**Note:** Unlike typical log directories, this folder and its contents are intentionally tracked in git to maintain a historical record of evaluation runs.

### Features

- Measures factual accuracy of research responses
- Uses GPT-4 as a grading model (configurable)
  ```python
  # In run_eval.py, you can customize the grader model:
  grader_model = ChatOpenAI(
      temperature=0,                           # Lower temperature for more consistent grading
      model_name="gpt-4-turbo",               # Can be changed to other OpenAI models
      openai_api_key=os.getenv("OPENAI_API_KEY")
  )
  ```
- Grades responses on a three-point scale:
  - `CORRECT`: Answer fully contains important information without contradictions
  - `INCORRECT`: Answer contains factual contradictions
  - `NOT_ATTEMPTED`: Answer neither confirms nor contradicts the target

**Note on Grader Configuration:** While the default grader uses GPT-4-turbo, you can modify the model and its parameters to use different OpenAI models or adjust the temperature for different grading behaviors. This is independent of the researcher's configuration, allowing you to optimize for cost or performance as needed.

### Metrics Tracked

- Accuracy rate
- F1 score
- Cost per query
- Success/failure rates
- Answer attempt rates
- Source coverage

### Running Evaluations

1. Install dependencies:
```bash
cd evals/simple_evals
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```bash
# Use the root .env file
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
LANGCHAIN_API_KEY=your_langchain_key_here
```

3. Run evaluation:
```bash
python run_eval.py --num_examples <number>
```

The `num_examples` parameter determines how many random test queries to evaluate (default: 1).

#### Customizing Researcher Behavior

The evaluation uses GPTResearcher with default settings, but you can modify `run_eval.py` to customize the researcher's behavior:

```python
researcher = GPTResearcher(
    query=query,
    report_type=ReportType.ResearchReport.value,  # Type of report to generate
    report_format="markdown",                      # Output format
    report_source=ReportSource.Web.value,         # Source of research
    tone=Tone.Objective,                          # Writing tone
    verbose=True                                  # Enable detailed logging
)
```

These parameters can be adjusted to evaluate different research configurations or output formats. For a complete list of configuration options, see the [configuration documentation](https://docs.gptr.dev/docs/Auto_Research_Engine/gptr/config).

**Note on Configuration Independence:** The evaluation system is designed to be independent of the researcher's configuration. This means you can use different LLMs and settings for evaluation versus research. For example:
- Evaluation could use GPT-4-turbo for grading while the researcher uses Claude 3.5 Sonnet for research
- Different retrievers, embeddings, or report formats can be used
- Token limits and other parameters can be customized separately

This separation allows for unbiased evaluation across different researcher configurations. However, please note that this feature is currently experimental and needs further testing.

### Output

The evaluation provides detailed metrics including:
- Per-query results with sources and costs
- Aggregate metrics (accuracy, F1 score)
- Total and average costs
- Success/failure counts
- Detailed grading breakdowns

### Example Output
```
=== Evaluation Summary ===
=== AGGREGATE METRICS ===

Debug counts:
Total successful: 100
CORRECT: 92
INCORRECT: 7
NOT_ATTEMPTED: 1
{
  "correct_rate": 0.92,
  "incorrect_rate": 0.07,
  "not_attempted_rate": 0.01,
  "answer_rate": 0.99,
  "accuracy": 0.9292929292929293,
  "f1": 0.9246231155778895
}
========================
Accuracy: 0.929
F1 Score: 0.925

Total cost: $1.2345
Average cost per query: $0.1371
``` 

## Hallucination Evaluation (`hallucination_eval/`)

The `hallucination_eval` directory contains tools for evaluating Auto_Research_Engine's outputs for hallucination. This evaluation system compares the generated research reports against their source materials to detect non-factual or hallucinated content, ensuring the reliability and accuracy of the research outputs.

### Components

- `run_eval.py`: Script to execute evaluations against Auto_Research_Engine
- `evaluate.py`: Core evaluation logic for detecting hallucinations
- `inputs/`: Directory containing test queries
  - `search_queries.jsonl`: Collection of research queries for evaluation
- `results/`: Directory containing evaluation results
  - `evaluation_records.jsonl`: Detailed per-query evaluation records
  - `aggregate_results.json`: Summary metrics across all evaluations

### Features

- Evaluates research reports against source materials
- Provides detailed reasoning for hallucination detection

### Running Evaluations

1. Install dependencies:
```bash
cd evals/hallucination_eval
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```bash
# Use the root .env file
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

3. Run evaluation:
```bash
python run_eval.py -n <number_of_queries>
```

The `-n` parameter determines how many queries to evaluate from the test set (default: 1).

### Example Output
```json
{
  "total_queries": 1,
  "successful_queries": 1,
  "total_responses": 1,
  "total_evaluated": 1,
  "total_hallucinated": 0,
  "hallucination_rate": 0.0,
  "results": [
    {
      "input": "What are the latest developments in quantum computing?",
      "output": "Research report content...",
      "source": "Source material content...",
      "is_hallucination": false,
      "confidence_score": 0.95,
      "reasoning": "The summary accurately reflects the source material with proper citations..."
    }
  ]
}
```

## Evaluating Generated Documents

Long-form reports should not be evaluated as a single monolithic answer. For this system, the recommended evaluation unit is:

1. Document
2. Section
3. Claim
4. Citation and source

This decomposition matches how Auto_Research_Engine works internally: the planner creates `section_details`, section research produces evidence packets, `check_data` scores evidence sufficiency, and the writer composes the final report.

### Recommended Document-Level Metrics

#### 1. Accuracy

For generated reports, `accuracy` is best measured at the claim level instead of the full-document level.

```text
claim_accuracy = supported_claims / judged_claims
```

Where:

- `supported_claims`: claims that are fully supported by the collected source text
- `judged_claims`: claims that were actually reviewed by an evaluator

Use this when you want a direct factual-correctness metric for long-form output.

#### 2. F1 Score

For reports, F1 is only meaningful when you have a gold claim set or a gold report outline to compare against.

```text
precision = matched_supported_claims / predicted_claims
recall = matched_gold_claims / gold_claims
f1 = 2 * precision * recall / (precision + recall)
```

If you do not have gold claims, do not force a document-level F1. In that case, claim accuracy, hallucination rate, citation quality, and completeness are usually more actionable.

#### 3. Hallucination Rate

For research reports, the most practical definition is claim-level unsupported content:

```text
hallucination_rate = unsupported_claims / total_claims
```

Where an `unsupported_claim` is a factual statement that cannot be grounded in the retrieved source text for that run.

This is stricter and more useful than only labeling an entire report as hallucinated or not hallucinated.

#### 4. Citation Coverage

Citation coverage measures how much of the report's factual content is explicitly backed by citations.

```text
citation_coverage = claims_with_citation / claims_requiring_citation
supported_citation_coverage = claims_with_valid_supporting_citation / claims_requiring_citation
```

Recommended practice:

- Track `citation_coverage` to measure whether claims are cited at all
- Track `supported_citation_coverage` to measure whether the attached citation actually supports the claim
- Optionally add `dead_link_rate` and `orphan_citation_rate`
- If you already have online artifacts such as `claim_annotations` or `claim_confidence_report`, still recompute these metrics offline from the rendered report and retrieved sources so the evaluation remains independent

#### 5. Source Diversity

Source diversity should not be measured by raw URL count alone. At minimum, track:

- `unique_domains`: number of distinct domains cited
- `top1_domain_share`: share of citations coming from the most-used domain
- `source_type_count`: number of source categories represented, such as academic papers, official filings, government sites, news, or company documentation
- `domain_entropy`: entropy over the domain distribution

This helps detect reports that look well-sourced but are actually over-dependent on one site or one type of evidence.

#### 6. Report Completeness

Completeness measures whether the generated report covered the planned structure and expected research scope.

Recommended metrics:

```text
section_completion = completed_sections / planned_sections
query_coverage = covered_research_queries / planned_research_queries
keypoint_coverage = covered_key_points / planned_key_points
```

A simple completeness score can be:

```text
completeness = 0.4 * section_completion
             + 0.3 * query_coverage
             + 0.3 * keypoint_coverage
```

### How to Apply These Metrics in This Repository

This repository already exposes several useful artifacts for document evaluation:

- `evals/hallucination_eval/` provides report-vs-source groundedness checks
- `multi_agents/agents/check_data.py` emits section-level `final_score`, `section_coverage`, and `ACCEPT/RETRY/BLOCKED`
- `multi_agents/agents/claim_verifier.py` emits online `source_index`, `claim_annotations`, and `claim_confidence_report`, which are useful debugging signals but should not replace independent offline evaluation
- `backend/server/server_utils.py` writes run logs containing `query`, `sources`, `context`, `report`, and `costs`
- `multi_agents/workflow_session.py` records workflow checkpoints for document and section replay

In practice, a strong evaluation pass for generated reports looks like this:

1. Run the system on a fixed report task set
2. Extract sections, claims, citations, and cited sources from each generated report
3. Compare claims against the collected source text
4. Compare section output against planner intent (`section_details`, `research_queries`, `key_points`)
5. Aggregate document-level metrics across the benchmark set

### Recommended Scorecard for Generated Reports

If you need a single summary view, use a scorecard that separates groundedness, evidence quality, and structural coverage:

```text
groundedness = 1 - hallucination_rate
citation_quality = supported_citation_coverage
completeness = 0.5 * section_completion
             + 0.3 * keypoint_coverage
             + 0.2 * query_coverage

overall = 0.4 * groundedness
        + 0.2 * citation_quality
        + 0.25 * completeness
        + 0.15 * source_diversity_score
```

In production, it is usually better to enforce hard thresholds than to rely only on a blended score. For example:

- `hallucination_rate <= x`
- `supported_citation_coverage >= y`
- `section_completion >= z`
- `top1_domain_share <= k`

### Important Limitations

- Hallucination judgments depend on source-text completeness. Missing evidence can produce false positives.
- F1 is only meaningful when you have a gold target to compare against.
- Citation coverage does not guarantee factual correctness unless the citation is checked for actual support.
- Diversity metrics are quality signals, not truth signals. A diverse set of bad sources is still bad.
