 # Prediction vs Causality: Why Your ML Model Can't Tell You What to Do

A data science notebook demonstrating one of the most important — and most 
commonly violated — boundaries in empirical research: the difference between 
prediction and causal inference.

## The Question

Education is the #1 feature in a gradient boosting model predicting wages, 
with 74.6% feature importance. Does that mean sending people to college 
will raise their wages by that amount?

**No. This notebook shows exactly why — and what to do instead.**

## What You'll Learn

- Why high feature importance does not imply a causal effect
- How omitted variable bias inflates education's apparent impact on wages
- Why adding observable controls often isn't enough
- How propensity score matching works and where it hits its limits
- When to use ML vs OLS vs matching vs instrumental variables

## The Data

**Current Population Survey (CPS) ASEC 2023** via [IPUMS CPS](https://cps.ipums.org)  
32,613 prime-age (25–55) wage and salary workers in the United States.

Key variables: hourly wage (constructed), education, experience, gender, 
race, ethnicity, marital status.

## Structure

| Section | What it covers |
|---|---|
| `01_eda` | Who earns what? Wage distributions, education gradient, gender/race gaps |
| `02_ml_model` | Gradient boosting model — R²=0.30, education dominates feature importances |
| `03_confounding` | Why the ML answer misleads — OLS stability and the DAG |
| `04_matching` | Propensity score matching — and why it hits the limits of observational data |
| `05_conclusion` | Practical decision guide: when to use which method |

## Key Finding

| Method | Education estimate | Limitation |
|---|---|---|
| ML feature importance | 74.6% importance score | Not a causal quantity |
| Naive OLS | 11.6% per year | Omitted variable bias |
| OLS with controls | 12.4% per year | Unobservables remain |
| Propensity score matching | 55.1% BA premium | Balance fails on unobservables |
| IV / RDD (literature) | ~7–10% per year | Requires exogenous variation |

The raw Bachelor's wage premium (55%) is real — but it reflects *who goes to 
college*, not just *what college does*. The causal return is likely far smaller.

## How to Run
```bash
git clone https://github.com/YOUR_USERNAME/prediction-vs-causality
cd prediction-vs-causality
pip install -r requirements.txt
jupyter notebook
```

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy
jupyter
```

## Background

This notebook was built as part of a broader project applying econometric 
thinking to modern ML workflows. The author is a PhD student in Economics 
at the University of Illinois Chicago with a background in program evaluation 
and causal inference.

Key references:
- Card, D. (1995). Using geographic variation in college proximity to estimate 
  the return to schooling.
- Angrist, J. & Krueger, A. (1991). Does compulsory school attendance affect 
  schooling and earnings?
- Cunningham, S. (2021). [Causal Inference: The Mixtape](https://mixtape.scunning.com) (free online)
- Huntington-Klein, N. (2021). [The Effect](https://theeffectbook.net) (free online)

---
*If this was useful, the two free textbooks linked above are the best next step 
for learning causal inference properly.*
```

Then create `requirements.txt`:
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
statsmodels>=0.14
scipy>=1.11
jupyter>=1.0