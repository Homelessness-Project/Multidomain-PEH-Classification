# Suggestions for IJCAI AI and Social Good Paper

Based on the comprehensive analyses performed, here are key findings and visualizations that would be strong for an IJCAI AI and Social Good paper:

## 1. Core Contribution: Understanding Discourse Patterns on Homelessness

### Key Findings to Highlight:

**A. Category Importance (Prevalence × Score)**
- Shows what's both common AND highly engaged with on Reddit
- Top categories: "express their opinion" (84.31% prevalence, 12.80 mean score)
- Reveals that opinion expression and societal critique dominate discourse
- **Social Good Angle**: Understanding what gets attention helps identify what narratives need intervention

**B. Statistical Significance with Bonferroni Correction**
- Rigorous statistical analysis accounting for multiple comparisons
- Identifies which categories are significantly different from others
- Demonstrates methodological rigor appropriate for AI research

**C. Score-Weighted vs Unweighted Analysis**
- Shows that certain category pairs are more common in highly-upvoted comments
- Reveals which narratives get amplified (e.g., "not in my backyard" has +6.02 higher scores)
- **Social Good Angle**: Identifies which harmful narratives are being promoted by engagement

## 2. Recommended Figures for Paper

### Figure 1: Category Importance (All 16 Categories)
- **File**: `category_importance_all16_with_errors.pdf`
- **Why**: Shows comprehensive view with error bars and Bonferroni correction
- **Caption**: "Category importance on Reddit (Prevalence × Mean Score) for all 16 categories. Error bars represent standard errors. Significance markers indicate Bonferroni-corrected p-values (* p<0.05, ** p<0.01, *** p<0.001). Green bars indicate categories significantly different from others."

### Figure 2: Prevalence and Mean Score Separately
- **File**: `category_prevalence_and_score_separate.pdf`
- **Why**: Shows the two components separately before combining
- **Caption**: "Category prevalence (left) and mean comment scores (right) for all 16 categories. Dashed lines indicate overall averages. Error bars represent standard errors."

### Figure 3: Score Differences by Category
- **File**: `category_score_analysis.pdf` (from previous analysis)
- **Why**: Shows which categories get higher/lower scores
- **Caption**: "Comment score differences by category. Categories with significantly higher scores (e.g., 'not in my backyard', 'harmful generalization') indicate narratives that receive more engagement."

### Figure 4: Co-occurrence Importance Heatmap
- **File**: `category_cooccurrence_importance_heatmap.pdf`
- **Why**: Shows which category pairs are most important together
- **Caption**: "Co-occurrence importance (Prevalence × Mean Score when co-occurring) for top 10 categories. Darker colors indicate higher importance."

## 3. Key Insights for Social Good

### A. Harmful Narratives Get Amplified
- Categories like "not in my backyard" (+6.02 score difference) and "harmful generalization" (+4.44) receive significantly more upvotes
- **Implication**: Platform algorithms may be amplifying harmful narratives
- **Action**: Need for content moderation and algorithmic interventions

### B. Rare Categories Are Invisible
- "Racist" category appears in only 0.16% of comments but has very low scores (5.53 mean)
- **Implication**: Extremely harmful content may be rare but still present
- **Action**: Need for specialized detection methods for rare but critical categories

### C. City Size Differences
- Large cities show more "government critique" (+8.34% difference)
- Small cities show more "provide an observation" (+6.08% difference)
- **Implication**: Different contexts require different intervention strategies
- **Action**: Context-aware interventions based on city characteristics

### D. Co-occurrence Patterns
- Strong co-occurrence between "express their opinion" and "societal critique" (Jaccard=0.7842)
- **Implication**: Opinions and critiques are tightly linked in discourse
- **Action**: Interventions should address both simultaneously

## 4. Methodological Contributions

### A. Importance Metric (Prevalence × Score)
- Novel metric combining frequency and engagement
- More informative than prevalence or score alone
- Captures what's actually prominent on the platform

### B. Statistical Rigor
- Bonferroni correction for multiple comparisons
- Error bars using proper error propagation
- Non-parametric tests (Mann-Whitney U) for score comparisons

### C. Multi-level Analysis
- Individual category analysis
- Co-occurrence analysis
- City size comparisons
- Score quartile analysis

## 5. Paper Structure Suggestions

### Abstract
- Focus on: Understanding discourse patterns on homelessness using AI/NLP
- Highlight: Novel importance metric and statistical rigor
- Emphasize: Social good implications (identifying harmful narratives, informing interventions)

### Introduction
- Problem: Homelessness discourse often contains harmful narratives
- Gap: Need for systematic analysis of what gets attention
- Contribution: Comprehensive analysis with novel metrics

### Methods
- Data: Reddit comments from 10 cities (36,288 comments)
- Categories: 16 categories from multi-label classification
- Metrics: Prevalence, Mean Score, Importance (Prevalence × Score)
- Statistics: Bonferroni correction, error propagation, Mann-Whitney U tests

### Results
- Section 1: Category importance (what's common AND highly engaged)
- Section 2: Score differences (what gets amplified)
- Section 3: Co-occurrence patterns (what appears together)
- Section 4: City size differences (contextual variations)

### Discussion
- Social Good Implications:
  1. Harmful narratives get amplified (need for algorithmic interventions)
  2. Rare categories need specialized detection
  3. Context matters (city size differences)
  4. Co-occurrence patterns suggest intervention strategies
- Limitations: Reddit-specific, may not generalize to other platforms
- Future Work: Intervention strategies, real-time monitoring

### Conclusion
- Key finding: Certain harmful narratives receive disproportionate engagement
- Impact: Informs content moderation and intervention strategies
- Vision: AI for understanding and mitigating harmful discourse

## 6. Additional Analyses to Consider

### A. Temporal Analysis
- How do patterns change over time?
- Are there trends in harmful narrative prevalence?

### B. Intervention Simulation
- What would happen if certain narratives were downranked?
- How would engagement patterns shift?

### C. Cross-Platform Comparison
- Compare Reddit patterns to X/Twitter, News, Meeting Minutes
- Identify platform-specific patterns

### D. Sentiment Analysis Integration
- Combine category analysis with sentiment
- Identify which categories are associated with negative sentiment

## 7. Key Messages for Social Good

1. **Transparency**: Making visible what narratives get attention
2. **Accountability**: Holding platforms accountable for amplifying harmful content
3. **Intervention**: Providing evidence-based guidance for content moderation
4. **Equity**: Identifying disparities in how different narratives are treated
5. **Prevention**: Early detection of harmful narrative patterns

## 8. Potential Paper Title

- "Understanding and Mitigating Harmful Narratives in Homelessness Discourse: An AI-Driven Analysis of Reddit Engagement Patterns"
- "What Gets Amplified? A Statistical Analysis of Discourse Patterns on Homelessness Using AI"
- "From Prevalence to Impact: A Novel Metric for Understanding Harmful Narratives in Social Media Discourse"

## 9. Ethical Considerations to Address

- **Privacy**: All data is de-identified
- **Bias**: Acknowledge potential biases in Reddit user base
- **Harm**: Analysis aims to reduce harm, not amplify it
- **Transparency**: Open methodology and reproducible analysis

## 10. Reproducibility

- All code and data processing steps documented
- Statistical methods clearly explained
- All visualizations generated from code (not manual)
- CSV files available for verification
