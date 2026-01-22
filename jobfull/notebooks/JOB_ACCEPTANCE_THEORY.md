# Job Acceptance Probability Predictor
## Using Bayesian Inference and Central Limit Theorem

This project aims to predict the likelihood of getting accepted for a CS job in Montreal, Canada, using data from job platforms like LinkedIn, Indeed, Glassdoor, etc.

---

## Dataset Sources

### 1. **Kaggle Datasets**
- [LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
  - 33K+ job listings with company details, skills, industries
- [Data Science Job Postings & Skills](https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills)
  - Job postings with required skills, experience levels
- [Indeed Job Postings](https://www.kaggle.com/search?q=indeed+jobs)
  - Multiple datasets with job descriptions, requirements

### 2. **Government/Public Data**
- [Statistics Canada - Labour Force Survey](https://www150.statcan.gc.ca/n1/en/type/data)
  - Employment rates, industry trends in Canada
- [Emploi-Québec](https://www.emploiquebec.gouv.qc.ca/citoyens/obtenir-information-marche-travail/)
  - Montreal-specific job market data

### 3. **Web Scraping (Legal/Ethical)**
- [Common Crawl](https://commoncrawl.org/)
  - Job board data snapshots
- Public APIs:
  - LinkedIn Jobs API (requires access)
  - Indeed API
  - GitHub Jobs (archived but data available)

### 4. **Academic Datasets**
- [UCI Machine Learning Repository - Job Recommendation](https://archive.ics.uci.edu/)
- Research papers often include supplementary job market datasets

---

## Problem Formulation

### Goal
Given an applicant's profile **P** = {skills, experience, education, location}, compute:

$$P(\text{Acceptance} \mid P) = \text{Probability of getting accepted}$$

### Key Features to Extract
1. **Skills** (e.g., Python, Java, Machine Learning)
2. **Years of Experience** (0-2, 3-5, 6-10, 10+)
3. **Education Level** (Bachelor's, Master's, PhD)
4. **Job Type** (Full-time, Contract, Internship)
5. **Company Size** (Startup, Mid-size, Enterprise)
6. **Location Match** (Montreal-based vs Remote)
7. **Application Response Rate** (if available from historical data)

---

## Theoretical Framework

### Phase A: Bayesian Inference Approach

We model the problem using **Naive Bayes Classification**:

#### Bayes' Theorem
$$P(\text{Accepted} \mid P) = \frac{P(P \mid \text{Accepted}) \cdot P(\text{Accepted})}{P(P)}$$

Where:
- $P(\text{Accepted} \mid P)$: Posterior probability of acceptance given your profile
- $P(P \mid \text{Accepted})$: Likelihood of your profile among accepted candidates
- $P(\text{Accepted})$: Prior probability of acceptance (base acceptance rate)
- $P(P)$: Evidence (normalizing constant)

#### Naive Bayes Assumption
Assume features are conditionally independent:

$$P(P \mid \text{Accepted}) = \prod_{i=1}^{n} P(f_i \mid \text{Accepted})$$

Where $f_i$ represents individual features (skills, experience, etc.)

#### With Laplace Smoothing
To avoid zero-probability trap when a feature hasn't been observed:

$$P(f_i \mid \text{Accepted}) = \frac{\text{count}(f_i, \text{Accepted}) + \alpha}{\text{count}(\text{Accepted}) + \alpha \cdot V}$$

Where:
- $\alpha$: Pseudo-count (typically 1)
- $V$: Number of unique values for feature $f_i$

---

### Phase B: Central Limit Theorem Application

#### Use Case 1: Estimating Population Acceptance Rate

If we collect sample data of $n$ job applications, the **sample mean** acceptance rate $\bar{X}$ follows:

$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$

Where:
- $\mu$: True population acceptance rate
- $\sigma^2$: Variance in acceptance rates
- $n$: Sample size

**Confidence Interval for Acceptance Rate:**
$$\mu \in \left[\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}}, \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right]$$

For 95% confidence: $z_{0.025} = 1.96$

#### Use Case 2: Aggregating Multiple Job Platform Scores

If you apply to $n$ jobs across different platforms, each with probability $p_i$:

$$\text{Expected Acceptances} = \sum_{i=1}^{n} p_i$$

By CLT, for large $n$:
$$\sum_{i=1}^{n} X_i \sim N\left(\sum_{i=1}^{n} p_i, \sum_{i=1}^{n} p_i(1-p_i)\right)$$

Where $X_i \sim \text{Bernoulli}(p_i)$

This allows computing: **"What's the probability of getting at least 1 acceptance from 50 applications?"**

---

## Model Pipeline

### Step 1: Data Collection & Preprocessing
```
1. Load job posting datasets from Kaggle/Web scraping
2. Filter for Montreal-based CS jobs
3. Extract features: required_skills, experience, education, etc.
4. Label data (if historical acceptance data available)
   - Alternative: Use proxy labels (application response rate, interview invites)
```

### Step 2: Feature Engineering
```
1. Skills matching score: 
   score = (your_skills ∩ required_skills) / required_skills
   
2. Experience match:
   - Exact match: 1.0
   - Within 1 year: 0.8
   - Within 2 years: 0.5
   - Otherwise: 0.2
   
3. Education encoding:
   - Bachelor's: 1
   - Master's: 2
   - PhD: 3
```

### Step 3: Bayesian Model Training
```
For each feature f in [skills, experience, education, ...]:
    θ_accepted[f] = (count(f, accepted) + α) / (count(accepted) + α·V)
    θ rejected[f] = (count(f, rejected) + α) / (count(rejected) + α·V)

Prior:
    P(accepted) = count(accepted) / total_applications
    P(rejected) = count(rejected) / total_applications
```

### Step 4: Prediction
```
For new application profile P:
    log_prob_accepted = log(P(accepted)) + Σ log(P(f_i | accepted))
    log_prob_rejected = log(P(rejected)) + Σ log(P(f_i | rejected))
    
    If log_prob_accepted > log_prob_rejected:
        Predict: Accepted
        Probability = exp(log_prob_accepted) / 
                     (exp(log_prob_accepted) + exp(log_prob_rejected))
```

### Step 5: CLT-Based Confidence Estimation
```
Given n similar applications with probabilities [p₁, p₂, ..., pₙ]:
    
    μ = Σ pᵢ / n  (mean acceptance probability)
    σ² = Σ (pᵢ - μ)² / n  (variance)
    
    95% CI = [μ - 1.96·σ/√n, μ + 1.96·σ/√n]
    
    P(at least 1 acceptance) = 1 - Π(1 - pᵢ)
```

---

## Evaluation Metrics

1. **Accuracy**: Overall correctness
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Precision**: Of predicted acceptances, how many are correct?
   $$\text{Precision} = \frac{TP}{TP + FP}$$

3. **Recall**: Of actual acceptances, how many did we catch?
   $$\text{Recall} = \frac{TP}{TP + FN}$$

4. **F1-Score**: Harmonic mean of precision and recall
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

5. **Calibration**: Are predicted probabilities well-calibrated?
   - Plot predicted probability vs actual acceptance rate

---

## Limitations & Considerations

### Model Limitations
1. **Feature Independence**: Naive Bayes assumes features are independent, which may not hold (e.g., Master's degree often correlates with more experience)
2. **Cold Start Problem**: New skills or job categories not in training data
3. **Class Imbalance**: Rejection rate >> acceptance rate in real world
4. **Proxy Labels**: May need to use interview invites as proxy for "acceptance" if true acceptance data unavailable

### Data Challenges
1. **Privacy**: Individual application outcomes are private
2. **Selection Bias**: Available datasets may not represent Montreal market accurately
3. **Temporal Shift**: Job market changes over time (tech layoffs, hiring freezes)
4. **Company-Specific**: Each company has unique criteria not captured by general features

### CLT Applicability
1. **Sample Size**: CLT requires sufficient sample size (n > 30 typically)
2. **Independence**: Assumes applications are independent (may not hold if applying to same companies repeatedly)
3. **Distribution**: Works best when underlying distribution isn't heavily skewed

---

## Extensions & Improvements

1. **Logistic Regression**: Compare with Naive Bayes
2. **Feature Interactions**: Use decision trees to capture non-linear relationships
3. **Temporal Analysis**: Model how acceptance probability changes over application season
4. **Multi-Platform Ensemble**: Combine predictions from LinkedIn, Indeed, Glassdoor models
5. **Skill Embeddings**: Use word2vec/BERT to capture semantic similarity between skills
6. **Monte Carlo Simulation**: Simulate multiple application campaigns to estimate acceptance distribution

---

## Implementation Checklist

- [ ] Download and explore datasets from Kaggle
- [ ] Filter for Montreal CS jobs
- [ ] Perform EDA (skill distribution, experience requirements, etc.)
- [ ] Create training labels (acceptance/rejection or proxy)
- [ ] Implement Naive Bayes with Laplace smoothing
- [ ] Train model and evaluate on test set
- [ ] Implement CLT-based confidence intervals
- [ ] Visualize: Feature importance, calibration plots, acceptance probability distribution
- [ ] Compare with baseline (e.g., always predict mean acceptance rate)
- [ ] Test with your own profile

---

## References

- **Naive Bayes Classification**: Pattern Recognition and Machine Learning (Bishop, 2006)
- **Central Limit Theorem**: Introduction to Probability (Blitzstein & Hwang, 2019)
- **Job Market Analysis**: LinkedIn Workforce Report, Statistics Canada Labour Force Survey
