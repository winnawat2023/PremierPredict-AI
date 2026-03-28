# PremierPredict-AI: A Stacking Ensemble Approach for Predicting English Premier League Match Outcomes

**Authors:**  
Chaiyaporn Homtean (67130700346)  
Krittin Chaisuvirat (67130700357)  
Pawornwit Maneenet (67130700361)  

**Supervisor:** Punnarust Silparattanawong  
**Course:** SEA612 Artificial Intelligence Fundamentals  
**Date:** 29 March 2026  

---

### Abstract
Predicting the outcome of professional football matches remains a significant challenge due to the high variability of the game. This paper presents PremierPredict-AI, a machine learning system utilizing a Stacking Ensemble architecture to predict English Premier League (EPL) match results. By integrating features such as ELO ratings, Transfermarkt market values, and historical performance windows, the model achieves a classification accuracy of 55.3%, surpassing benchmarks like the FPL "Wisdom of Crowds" (54.7%) and BBC expert panels (47.1%).

---

## I. Introduction
The EPL is a data-rich environment where predicting outcomes (Home, Draw, Away) is a complex classification problem. Football is inherently chaotic, influenced by tactical shifts, injuries, and red cards. This paper documents PremierPredict-AI—a project for the SEA612 course that uses a Stacking Ensemble architecture to create a "team" of AI models that produce a final consensus.

## II. Literature Review
Historically, the **ELO Rating system** was the gold standard for skill ranking. However, it lacks the ability to account for sudden squad improvements. Recent studies show that **Market Value** (Transfermarkt) is a high-level predictor of long-term success. Our project combines Stats (ELO), Money (Market Value), and Human Intuition (FPL Crowds).

## III. Methodology & Development Roadmap
We followed an iterative 4-step development process:
1. **V1.0 Baseline (44.8%):** Initial model using simple statistics.
2. **V2.5 Feature Engineering (51.2%):** Added ELO and Market Value. Surpassed BBC Experts (47.1%).
3. **V3.8 Advanced Single Model (53.9%):** Switched to XGBoost. Robust but still lost to FPL fans (54.7%).
4. **V5.0 Stacking Ensemble (55.3%):** Final version using a hierarchical approach to beat all human benchmarks.

### Stacking Architecture:
- **Layer 1 (Base):** XGBoost, Random Forest, and Gradient Boosting.
- **Layer 2 (Meta):** Logistic Regression acts as the "Judge" to decide which model to trust for specific match conditions.

## IV. Experimental Results

| Forecaster | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| BBC Expert (Chris Sutton) | 47.1% | 0.48 |
| FPL "Wisdom of Crowds" | 54.7% | 0.52 |
| Single XGBoost Model | 53.9% | 0.54 |
| **PremierPredict-AI V5.0** | **55.3%** | **0.56** |

### Top Features (Correlation to Outcome):
- **ELO Differential:** 0.61 (High)
- **Market Value Diff:** 0.42 (Medium)
- **Home Field Advantage:** 0.32 (Medium)
- **FPL Captaincy (Sentiment):** 0.27 (Low)

## V. Discussion & Challenges
- **The Draw Dilemma:** Draws are the hardest to predict (51.2% precision vs 71.3% for wins) as they often result from random mid-match events.
- **Biased Big Teams:** Early versions were too biased toward "big clubs." We solved this with Probability Calibration, making the model more "honest" about its confidence.

## VI. Conclusion
PremierPredict-AI successfully bridges historical stats and financial market valuations to beat human benchmarks. For SEA612, this proves that AI can systematically remove emotional bias from decision-making, revealing underlying patterns in the "Beautiful Game."

---
## References
[1] D. Berrar, "Tree-based Ensembles for Sports Analytics," 2022.  
[2] T. Peeters, "Estimating the Economic Value of EPL Squads," 2023.  
[3] Football-Data.co.uk Match Database, 2024.  
[4] ClubELO Rankings, 2024.
