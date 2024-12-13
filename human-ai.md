# Human_AI


## Introduction

This analysis examines the conditions under which human-AI collaboration
(HAI) outperforms AI alone in decision tasks. We investigate various
factors including AI explanations, user expertise, and AI confidence
levels.

``` r
# Load required libraries
library(tidyverse)
library(brms)
library(tidybayes)
library(knitr)
library(ggplot2)
library(corrplot)
library(effectsize)
library(interactions)

# Set seed for reproducibility
set.seed(42)
```

## Data Preparation

``` r
# Load and prepare data
df <- read_csv("Data_Extraction (1).csv") %>%
  # Filter for decision tasks
  filter(Task_Type == "Decide") %>%
  # Calculate performance metrics
  mutate(
    # Calculate HAI advantage over AI
    hai_advantage = Avg_Perf_HumanAI_Adj - Avg_Perf_AI_Adj,
    # Binary indicator for cases where HAI beats AI
    hai_beats_ai = hai_advantage > 0,
    # Convert relevant columns to logical
    AI_Expl_Incl = if_else(AI_Expl_Incl == "Yes", TRUE, FALSE),
    AI_Conf_Incl = if_else(AI_Conf_Incl == "Yes", TRUE, FALSE),
    Participant_Expert = if_else(Participant_Expert == "Yes", TRUE, FALSE)
  )
```

    Rows: 370 Columns: 64
    ── Column specification ────────────────────────────────────────────────────────
    Delimiter: ","
    chr (41): Paper_Name, ES_ID, Title, Authors, Venue, Exp_Design, Comp_Type, T...
    dbl (23): Paper_ID, Exp_ID, Treatment_ID, Measure_ID, Exp_ID_Cleaned, Year, ...

    ℹ Use `spec()` to retrieve the full column specification for this data.
    ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# Create dataset of synergy cases
synergy_cases <- df %>%
  filter(hai_beats_ai)
```

## Overview of Synergy Cases

``` r
# Basic Summary Statistics
synergy_summary <- data.frame(
  Metric = c("Total Decision Tasks", 
             "Cases with HAI > AI",
             "Proportion with Synergy",
             "Mean HAI Advantage (when present)",
             "SD of HAI Advantage (when present)"),
  Value = c(nrow(df),
            nrow(synergy_cases),
            round(nrow(synergy_cases) / nrow(df), 3),
            round(mean(synergy_cases$hai_advantage), 3),
            round(sd(synergy_cases$hai_advantage), 3))
)

kable(synergy_summary, caption = "Overview of Synergy Cases")
```

| Metric                             |    Value |
|:-----------------------------------|---------:|
| Total Decision Tasks               |  336.000 |
| Cases with HAI \> AI               |  164.000 |
| Proportion with Synergy            |    0.488 |
| Mean HAI Advantage (when present)  |  110.211 |
| SD of HAI Advantage (when present) | 1198.294 |

Overview of Synergy Cases

## Factor Analysis

``` r
# Enhanced Factor Analysis with Effect Sizes
factor_comparison <- df %>%
  group_by(hai_beats_ai) %>%
  summarise(
    n_cases = n(),
    pct_with_explanations = mean(AI_Expl_Incl) * 100,
    pct_with_confidence = mean(AI_Conf_Incl) * 100,
    pct_expert_users = mean(Participant_Expert) * 100,
    mean_ai_accuracy = mean(Avg_Perf_AI_Adj),
    sd_ai_accuracy = sd(Avg_Perf_AI_Adj),
    .groups = 'drop'
  )

# Calculate effect sizes
effect_sizes <- data.frame(
  Factor = c("AI Explanations", "AI Confidence", "Expert Users", "AI Accuracy"),
  Cohens_d = c(
    cohens_d(as.numeric(AI_Expl_Incl) ~ hai_beats_ai, data = df)$Cohens_d,
    cohens_d(as.numeric(AI_Conf_Incl) ~ hai_beats_ai, data = df)$Cohens_d,
    cohens_d(as.numeric(Participant_Expert) ~ hai_beats_ai, data = df)$Cohens_d,
    cohens_d(Avg_Perf_AI_Adj ~ hai_beats_ai, data = df)$Cohens_d
  )
)
```

    Warning: 'y' is numeric but has only 2 unique values.
      If this is a grouping variable, convert it to a factor.
    Warning: 'y' is numeric but has only 2 unique values.
      If this is a grouping variable, convert it to a factor.
    Warning: 'y' is numeric but has only 2 unique values.
      If this is a grouping variable, convert it to a factor.

``` r
# Print tables
kable(factor_comparison, 
      caption = "Comparison of Factors Between Synergy and Non-Synergy Cases",
      digits = 2)
```

| hai_beats_ai | n_cases | pct_with_explanations | pct_with_confidence | pct_expert_users | mean_ai_accuracy | sd_ai_accuracy |
|:---|---:|---:|---:|---:|---:|---:|
| FALSE | 172 | 47.67 | 35.47 | 26.16 | -8791.45 | 38021.40 |
| TRUE | 164 | 46.34 | 44.51 | 26.83 | -1988.20 | 17953.91 |

Comparison of Factors Between Synergy and Non-Synergy Cases

``` r
kable(effect_sizes,
      caption = "Effect Sizes for Key Factors",
      digits = 3)
```

| Factor          | Cohens_d |
|:----------------|---------:|
| AI Explanations |    0.027 |
| AI Confidence   |   -0.185 |
| Expert Users    |   -0.015 |
| AI Accuracy     |   -0.227 |

Effect Sizes for Key Factors

## Correlation Analysis

``` r
# Create correlation matrix
cor_matrix <- df %>%
  select(hai_advantage, AI_Expl_Incl, AI_Conf_Incl, 
         Participant_Expert, Avg_Perf_AI_Adj) %>%
  cor()

# Plot correlation matrix
corrplot(cor_matrix, 
         method = "color",
         type = "upper",
         addCoef.col = "black",
         tl.col = "black",
         tl.srt = 45,
         diag = FALSE)
```

![Correlation Matrix of Key
Factors](human-ai_files/figure-commonmark/correlation-1.png)

## Hierarchical Bayesian Analysis

``` r
# Fit hierarchical Bayesian model
model_synergy <- brm(
  hai_advantage ~ AI_Expl_Incl * Participant_Expert +
                  AI_Conf_Incl * Avg_Perf_AI_Adj +
                  (1|Paper_ID),
  data = df,
  family = gaussian(),
  prior = c(
    prior(normal(0, 0.5), class = "Intercept"),
    prior(normal(0, 0.5), class = "b"),
    prior(exponential(1), class = "sd")
  ),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 42
)
```

    Compiling Stan program...

    Start sampling

``` r
# Model Summary
summary(model_synergy)
```

     Family: gaussian 
      Links: mu = identity; sigma = identity 
    Formula: hai_advantage ~ AI_Expl_Incl * Participant_Expert + AI_Conf_Incl * Avg_Perf_AI_Adj + (1 | Paper_ID) 
       Data: df (Number of observations: 336) 
      Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
             total post-warmup draws = 4000

    Multilevel Hyperparameters:
    ~Paper_ID (Number of levels: 65) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     0.98      0.97     0.02     3.54 1.00     3309     1989

    Regression Coefficients:
                                            Estimate Est.Error l-95% CI u-95% CI
    Intercept                                1179.40     19.08  1141.14  1216.48
    AI_Expl_InclTRUE                           -0.01      0.49    -0.96     0.96
    Participant_ExpertTRUE                      0.01      0.50    -0.97     1.00
    AI_Conf_InclTRUE                           -0.00      0.50    -0.98     0.97
    Avg_Perf_AI_Adj                             0.27      0.00     0.27     0.28
    AI_Expl_InclTRUE:Participant_ExpertTRUE     0.00      0.50    -0.95     0.97
    AI_Conf_InclTRUE:Avg_Perf_AI_Adj           -0.33      0.01    -0.35    -0.31
                                            Rhat Bulk_ESS Tail_ESS
    Intercept                               1.00     5134     2847
    AI_Expl_InclTRUE                        1.00     5779     3037
    Participant_ExpertTRUE                  1.00     6190     2663
    AI_Conf_InclTRUE                        1.00     6102     2932
    Avg_Perf_AI_Adj                         1.00     5529     3509
    AI_Expl_InclTRUE:Participant_ExpertTRUE 1.00     6427     2859
    AI_Conf_InclTRUE:Avg_Perf_AI_Adj        1.00     6224     3413

    Further Distributional Parameters:
          Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sigma  1908.85     75.50  1766.35  2063.55 1.00     5452     2591

    Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
    and Tail_ESS are effective sample size measures, and Rhat is the potential
    scale reduction factor on split chains (at convergence, Rhat = 1).

## Posterior Probability Analysis

``` r
# Extract posterior probabilities
draws <- as_draws_df(model_synergy)
prob_positive <- data.frame(
  Factor = c("AI Explanations", 
             "Expert Users", 
             "AI Confidence", 
             "AI Performance",
             "AI Explanations:Expert Users",
             "AI Confidence:AI Performance"),
  Prob_Positive = c(
    mean(draws$b_AI_Expl_InclTRUE > 0),
    mean(draws$b_Participant_ExpertTRUE > 0),
    mean(draws$b_AI_Conf_InclTRUE > 0),
    mean(draws$b_Avg_Perf_AI_Adj > 0),
    mean(draws$`b_AI_Expl_InclTRUE:Participant_ExpertTRUE` > 0),
    mean(draws$`b_AI_Conf_InclTRUE:Avg_Perf_AI_Adj` > 0)
  )
)

kable(prob_positive, 
      caption = "Probability of Positive Effect on HAI Advantage",
      digits = 3)
```

| Factor                       | Prob_Positive |
|:-----------------------------|--------------:|
| AI Explanations              |         0.496 |
| Expert Users                 |         0.505 |
| AI Confidence                |         0.500 |
| AI Performance               |         1.000 |
| AI Explanations:Expert Users |         0.496 |
| AI Confidence:AI Performance |         0.000 |

Probability of Positive Effect on HAI Advantage

## Visualization of Effects

``` r
# Main effects plot
ggplot(df, aes(x = Avg_Perf_AI_Adj, y = hai_advantage, color = AI_Expl_Incl)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm") +
  facet_wrap(~Participant_Expert) +
  theme_minimal() +
  labs(
    title = "HAI Advantage vs AI Performance",
    subtitle = "Faceted by User Expertise, Colored by AI Explanations",
    x = "AI Performance",
    y = "HAI Advantage over AI",
    color = "AI Explanations\nIncluded"
  )
```

    `geom_smooth()` using formula = 'y ~ x'

![HAI Advantage vs AI Performance by User Expertise and AI
Explanations](human-ai_files/figure-commonmark/visualize-effects-1.png)

``` r
# Interaction plot
interact_plot(model_synergy, 
             pred = AI_Expl_Incl, 
             modx = Participant_Expert,
             plot.points = TRUE)
```

    ✖ Detected factor predictor.
    ℹ Plotting with cat_plot() instead.
    ℹ See `?interactions::cat_plot()` for full details on how to specify models
      with categorical predictors.
    ℹ If you experience errors or unexpected results, try using cat_plot()
      directly.

![Interaction Effects Between AI Explanations and User
Expertise](human-ai_files/figure-commonmark/interaction-plot-1.png)
