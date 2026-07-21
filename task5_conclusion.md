# Task 5: Conclusion (1 Mark)

The Multiple Linear Regression model predicts medical insurance charges with a reasonable
degree of accuracy, explaining about 78% of the variance in the test data (R² ≈ 0.78) with
an average error of roughly $4,180 (MAE). Among all predictors, **smoking status** is by far
the strongest driver of charges, followed by **BMI**, **age**, and **number of children**;
`sex` and `region` have only a marginal effect. These findings align with real-world
expectations — smokers and individuals with higher BMI generally incur higher medical costs.

One key **limitation of Linear Regression** here is its assumption of a purely linear,
additive relationship between features and charges: in reality, the effect of BMI and age on
cost is much steeper for smokers than for non-smokers (an interaction effect), and the
underlying cost distribution is skewed rather than normally distributed. Capturing this would
likely require interaction terms, log-transforming the target, or a non-linear model such as
a decision tree or gradient boosting regressor.
