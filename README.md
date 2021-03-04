# Conditional calibration of conformal p-values

We study the construction of p-values for nonparametric outlier detection, taking a multiple-testing perspective.  The framework is that of conformal prediction, which wraps around any machine-learning algorithm to provide finite-sample guarantees regarding the validity of predictions for future testpoints.  In this setting, existing methods can compute p-values that are marginally valid but mutually dependent for different future test points. 

This repository contains a software implementation and guided examples for the methodology developed in the following accompanying paper, which provides a new method to  compute p-values that are both conditionally valid and independent of each other for different future test points, thus allowing multiple testing with stronger stronger type-I error guarantees.
  ```
  "Multiple Outlier Testing with Conformal p-values"
  Stephen Bates, Emmanuel Candes, Lihua Lei, Yaniv Romano, and Matteo Sesia. 
  arXiv preprint (2021).
  ```
