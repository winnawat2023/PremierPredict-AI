![Modern Architecture](file:///Users/h.chaiyaporn/code/PremierPredict-AI/docs/architecture.png)

## 1. System Architecture
This infographic shows the end-to-end data flow with a modern, colorful design.

```mermaid
graph TD
    subgraph "Data Sources (External)"
        DS1[Football-Data.co.uk - Match History]
        DS2[ClubELO - Team Ratings]
        DS3[Transfermarkt - Market Values]
        DS4[FPL API - Fan Sentiment]
    end

    subgraph "Data Pipeline (Python/Pandas)"
        P1[Data Cleaning & Integration]
        P2[Feature Engineering: Rolling L5 Form]
        P3[ELO Probability Calculation]
        P4[Feature Scaling & Normalization]
    end

    subgraph "Model Layer (Stacking Ensemble)"
        B1[Base: XGBoost]
        B2[Base: Random Forest]
        B3[Base: Gradient Boosting]
        M1[Meta-Learner: Logistic Regression]
    end

    subgraph "Deployment & UI"
        D1[GitHub Repository]
        D2[Streamlit Community Cloud]
        U1[Interactive Dashboard UI]
    end

    %% Connections
    DS1 & DS2 & DS3 & DS4 --> P1
    P1 --> P2 --> P3 --> P4
    P4 --> B1 & B2 & B3
    B1 & B2 & B3 --> M1
    M1 --> D1
    D1 --> D2 --> U1
```

![Modern Flowchart](file:///Users/h.chaiyaporn/code/PremierPredict-AI/docs/flowchart.png)

---

## 2. Logic Flowchart (Prediction Flow)
This diagram shows the step-by-step logic for a single match prediction.

```mermaid
flowchart TD
    A[Start: Match Fixture Received] --> B[Fetch Historical Data]
    B --> C{Data Complete?}
    C -- No --> D[Use Imputation/Default Baseline]
    C -- Yes --> E[Compute Differentials: ELO & MV]
    D & E --> F[Run Level-0 Ensemble Prediction]
    F --> G[Extract Probability Features]
    G --> H[Run Level-1 Meta-Learner Decision]
    H --> I[Apply Probability Calibration]
    I --> J[Display Prediction & Confidence on Dashboard]
    J --> K[End]
```
