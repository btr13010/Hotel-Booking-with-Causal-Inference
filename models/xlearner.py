import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

class XLearner:
    def __init__(
        self,
        treatment,
        confounders,
        confounders_treated,
        confounders_control,
        outcome_treated,
        outcome_control,
    ):
        self.w_treated = confounders_treated
        self.w_control = confounders_control
        self.y_treated = outcome_treated
        self.y_control = outcome_control
        self.t = treatment
        self.w = confounders
    
    def estimate(self):
        # fit a logistic regression model to predict the outcome of the treated and control group
        model_treated = LogisticRegression().fit(self.w_treated, self.y_treated)
        model_control = LogisticRegression().fit(self.w_control, self.y_control)

        # use the fitted model for treated group to predict counterfactual outcome for the control group, and vice versa
        control_counterfactual = model_treated.predict(self.w_control)
        treated_counterfactual = model_control.predict(self.w_treated)

        # calculate the ITE for each population using the observed and counterfactual outcome
        treated_ite = self.y_treated - treated_counterfactual
        control_ite = control_counterfactual - self.y_control

        # fit a linear regression model to predict the ITE for the treated and control group
        treated_ite_model = LinearRegression().fit(self.w_treated, treated_ite)
        control_ite_model = LinearRegression().fit(self.w_control, control_ite)

        # use the fitted models to predict the intervened ite on the entire population for each case of treatment
        treated_ite_pred = treated_ite_model.predict(self.w)
        control_ite_pred = control_ite_model.predict(self.w)

        # calculate the propensity score 
        prop_model = LogisticRegression().fit(self.w, self.t)
        prop_score = prop_model.predict_proba(self.w)[:, 1]

        # calculate the final ate
        ite = prop_score * treated_ite_pred + (1 - prop_score) * control_ite_pred

        return ite.mean()