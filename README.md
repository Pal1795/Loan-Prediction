# Loan-Prediction

**Should this loan be APPROVED or DENIED?**

**Problem Statement:**
- Loan Default Prediction: Banks struggle to assess default probabilities, affecting decisions and profitability.
- Threshold Dilemma: Identifying the right probability cut off is key to balancing risks and maximizing profits.

**Solution:**
- Build predictive Machine learning models such as KNN, Classification trees, Logistic Regression, Neural Networks, Discriminant Analysis to predict loan outcomes.
- Determine the Optimal classification threshold to guide loan approval decisions, balancing risks and profitability to maximize the bank's net profit.

**Data Overview:**

Downloaded the SBAnational.csv data from Kaggle. It has 899,164 rows and 27 columns.

**Actionable Insights:**

- Loan Approval Strategy: Approve loans in the order of predicted probabilities, prioritizing the least risky applicants. This will increase the likelihood of maximizing profits while minimizing defaults.

- Optimal Loan Approval Threshold: Stop approving loans after the 147,072nd loan to achieve maximum profitability.

- Threshold for Future Approvals: For future loan approvals, use a probability threshold of 0.3804, which balances the trade-off between loan approvals and profitability.

**Key Results:**

- Maximum Net Profit: The model generated a maximum cumulative net profit of $1,589,851,091.60, achieved by approving 147,072 loans. This highlights the effectiveness of using machine learning models for optimizing profitability in financial decisions.

- Optimal Cut-Off Probability: The model’s optimal cut-off probability for approving loans is 0.3804. This threshold ensures the maximum net profit by approving loans with the lowest risk.

- Gains Chart: The cumulative profit steadily increases as more loans are approved, peaking when the optimal number of loans (147,072) is reached.

- Lift Chart: The Lift Chart illustrates the model's efficiency in prioritizing low-risk loans, enabling lenders to focus on applicants with the highest likelihood of repaying their loans.

**Conclusion:**

The analysis leverages the Random Forest Model to predict the risk associated with loan applications, classifying them into low-risk and high-risk categories. The model has demonstrated strong performance in accurately identifying risk levels, providing a valuable tool for decision-making in loan approval processes.
