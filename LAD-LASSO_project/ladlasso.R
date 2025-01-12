library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)

#double exp errror with n=50, sigma=1

n <- 50
p <- 8
sigma <- 1
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma*rlaplace(n, location = 0, scale = sigma)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)


##################

#double exp errror with n=100, sigma=1

n <- 100
p <- 8
sigma <- 1
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 100

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma*rlaplace(n, location = 0, scale = sigma)
  y <- y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)


##############
#double exp errror with n=200, sigma=1

library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 200
p <- 8
sigma <- 1
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rlaplace(n, location = 0, scale = sigma)
  y <-  0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)


#################
#double exp errror with n=50, sigma=0.5


library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 50
p <- 8
sigma <- 0.5
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rlaplace(n, location = 0, scale = sigma)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)


####################

#double exp errror with n=100, sigma=0.5

library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 100
p <- 8
sigma <- 0.5
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rlaplace(n, location = 0, scale = sigma)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)



####################

#double exp errror with n=200, sigma=0.5

library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 200
p <- 8
sigma <- 0.5
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rlaplace(n, location = 0, scale = sigma)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)


##################


####################

#t5 with n=50, sigma=1

library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 50
p <- 8
sigma <- 1
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rt(n, df=5)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)

##t(5) error n=100, sigma=1

library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 100
p <- 8
sigma <- 1
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rt(n, df = 5)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)

###
##t(5) error n=200 sigma =1

library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 200
p <- 8
sigma <- 1
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rt(n, df = 5)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)

##t(3) error sigma=1, n=50

library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 50
p <- 8
sigma <- 1
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rt(n, df = 3)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)

##t(3) error, n=100, sigma=1

library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 100
p <- 8
sigma <- 1
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rt(n, df = 3)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)


##t(3) error, sigma=1, n=200

library(MASS)          
library(LaplacesDemon) 
library(ncvreg)
library(glmnet)
library(regnet)


n <- 200
p <- 8
sigma <- 1
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rt(n, df = 3)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)


#########
#t3 error with sigma=0.5, n=50

n <- 50
p <- 8
sigma <- 0.5
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rt(n, df = 3)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)



############t3, sigma=0.5, n=100


n <- 100
p <- 8
sigma <- 0.5
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma*rt(n, df=3)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)


############



###t(3), n=200, sigma=0.5

n <- 200
p <- 8
sigma <- 0.5
beta <- c(0.5, 1.0, 1.5, 2.0, rep(0, p - 4))

mean_vec <- rep(0, p)
cov_mat <- diag(p)

classify_fit <- function(coef_est, true_beta) {
  true_nonzero <- which(true_beta != 0)
  true_zero <- which(true_beta == 0)
  
  coef_est_no_intercept <- coef_est[-1]
  
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  includes_all_nonzero <- all(true_nonzero %in% selected)
  includes_extra <- any(selected %in% true_zero)
  
  if (!includes_all_nonzero) {
    return("Underfitted")
  } else if (includes_all_nonzero && !includes_extra) {
    return("Correctly fitted")
  } else {
    return("Overfitted")
  }
}

calc_zero_stats <- function(coef_est, true_beta) {
  true_zero <- which(true_beta == 0)
  coef_est_no_intercept <- coef_est[-1]
  selected <- which(abs(coef_est_no_intercept) > 1e-8)
  
  correct_zeros <- length(setdiff(true_zero, selected))
  incorrect_zeros <- length(intersect(true_zero, selected))
  
  return(list(correct_zeros=correct_zeros, incorrect_zeros=incorrect_zeros))
}

mape_function <- function(y, y_hat) {
  mean(abs(y - y_hat))
}

n_reps <- 1000

methods <- c("SCAD","MCP","LASSO_glmnet","LASSO_regnet","ORACLE")

init_counts <- function() {
  list(
    under=0, correct=0, over=0,
    correct_zeros_vec=numeric(n_reps),
    incorrect_zeros_vec=numeric(n_reps),
    mape_vec=numeric(n_reps)
  )
}

scad_results <- init_counts()
mcp_results <- init_counts()
lasso_glmnet_results <- init_counts()
regnet_lasso_results <- init_counts()
oracle_results <- init_counts()

for (i in 1:n_reps) {
  x <- mvrnorm(n, mean_vec, cov_mat)
  errors <- sigma* rt(n, df = 3)
  y <- 0.5 * x[, 1] + 1 * x[, 2] + 1.5 * x[, 3] + 2 * x[, 4] + errors
  
  fit_scad <- cv.ncvreg(x, y, penalty = "SCAD")
  coef_scad <- coef(fit_scad, s = "lambda.min")
  res_scad <- classify_fit(coef_scad, beta)
  
  if (res_scad == "Underfitted") scad_results$under <- scad_results$under + 1
  if (res_scad == "Correctly fitted") scad_results$correct <- scad_results$correct + 1
  if (res_scad == "Overfitted") scad_results$over <- scad_results$over + 1
  
  scad_zeros_stats <- calc_zero_stats(coef_scad, beta)
  scad_results$correct_zeros_vec[i] <- scad_zeros_stats$correct_zeros
  scad_results$incorrect_zeros_vec[i] <- scad_zeros_stats$incorrect_zeros
  
  y_hat_scad <- predict(fit_scad, X=x, s="lambda.min")
  scad_results$mape_vec[i] <- mape_function(y, y_hat_scad)
  
  fit_mcp <- cv.ncvreg(x, y, penalty = "MCP")
  coef_mcp <- coef(fit_mcp, s = "lambda.min")
  res_mcp <- classify_fit(coef_mcp, beta)
  
  if (res_mcp == "Underfitted") mcp_results$under <- mcp_results$under + 1
  if (res_mcp == "Correctly fitted") mcp_results$correct <- mcp_results$correct + 1
  if (res_mcp == "Overfitted") mcp_results$over <- mcp_results$over + 1
  
  mcp_zeros_stats <- calc_zero_stats(coef_mcp, beta)
  mcp_results$correct_zeros_vec[i] <- mcp_zeros_stats$correct_zeros
  mcp_results$incorrect_zeros_vec[i] <- mcp_zeros_stats$incorrect_zeros
  
  y_hat_mcp <- predict(fit_mcp, X=x, s="lambda.min")
  mcp_results$mape_vec[i] <- mape_function(y, y_hat_mcp)
  
  fit_lasso_cv <- cv.glmnet(x, y, alpha = 1)
  coef_lasso <- coef(fit_lasso_cv, s = "lambda.min")
  res_lasso <- classify_fit(as.matrix(coef_lasso), beta)
  
  if (res_lasso == "Underfitted") lasso_glmnet_results$under <- lasso_glmnet_results$under + 1
  if (res_lasso == "Correctly fitted") lasso_glmnet_results$correct <- lasso_glmnet_results$correct + 1
  if (res_lasso == "Overfitted") lasso_glmnet_results$over <- lasso_glmnet_results$over + 1
  
  lasso_zeros_stats <- calc_zero_stats(as.matrix(coef_lasso), beta)
  lasso_glmnet_results$correct_zeros_vec[i] <- lasso_zeros_stats$correct_zeros
  lasso_glmnet_results$incorrect_zeros_vec[i] <- lasso_zeros_stats$incorrect_zeros
  
  y_hat_lasso <- predict(fit_lasso_cv, newx=x, s="lambda.min")
  lasso_glmnet_results$mape_vec[i] <- mape_function(y, y_hat_lasso)
  
  out_regnet <- cv.regnet(x, y, response="continuous", penalty="lasso", folds=5)
  fit_regnet <- regnet(x, y, response="continuous", penalty="lasso", out_regnet$lambda)
  
  coef_regnet_full <- c( fit_regnet$coeff)
  
  res_regnet <- classify_fit(coef_regnet_full, beta)
  
  if (res_regnet == "Underfitted") regnet_lasso_results$under <- regnet_lasso_results$under + 1
  if (res_regnet == "Correctly fitted") regnet_lasso_results$correct <- regnet_lasso_results$correct + 1
  if (res_regnet == "Overfitted") regnet_lasso_results$over <- regnet_lasso_results$over + 1
  
  regnet_zeros_stats <- calc_zero_stats(coef_regnet_full, beta)
  regnet_lasso_results$correct_zeros_vec[i] <- regnet_zeros_stats$correct_zeros
  regnet_lasso_results$incorrect_zeros_vec[i] <- regnet_zeros_stats$incorrect_zeros
  
  y_hat_regnet <- x %*% fit_regnet$coeff[-1]
  regnet_lasso_results$mape_vec[i] <- mape_function(y, y_hat_regnet)
  
  x_oracle <- x[, 1:4, drop=FALSE]
  out_oracle <- cv.regnet(x_oracle, y, response ="continuous", penalty="lasso", folds=5)
  fit_oracle <- regnet(x_oracle, y, response = "continuous", penalty="lasso",out_oracle$lambda)
  
  coef_oracle_full <- c(fit_oracle$coeff, rep(0, p - 4))
  res_oracle <- classify_fit(coef_oracle_full, beta)
  
  if (res_oracle == "Underfitted") oracle_results$under <- oracle_results$under + 1
  if (res_oracle == "Correctly fitted") oracle_results$correct <- oracle_results$correct + 1
  if (res_oracle == "Overfitted") oracle_results$over <- oracle_results$over + 1
  
  oracle_zeros_stats <- calc_zero_stats(coef_oracle_full, beta)
  oracle_results$correct_zeros_vec[i] <- oracle_zeros_stats$correct_zeros
  oracle_results$incorrect_zeros_vec[i] <- oracle_zeros_stats$incorrect_zeros
  
  y_hat_oracle <-  x_oracle %*% fit_oracle$coeff[-1]
  oracle_results$mape_vec[i] <- mape_function(y, y_hat_oracle)
}

summarize_results <- function(res_list, method_name) {
  under_pct <- res_list$under / n_reps 
  correct_pct <- res_list$correct / n_reps 
  over_pct <- res_list$over / n_reps 
  avg_correct_zeros <- mean(res_list$correct_zeros_vec)
  avg_incorrect_zeros <- mean(res_list$incorrect_zeros_vec)
  avg_mape <- mean(res_list$mape_vec)
  med_mape <- median(res_list$mape_vec)
  
  data.frame(
    Method = method_name,
    Underfitted = under_pct,
    Correctly_fitted = correct_pct,
    Overfitted = over_pct,
    Avg_Correct_Zeros = avg_correct_zeros,
    Avg_Incorrect_Zeros = avg_incorrect_zeros,
    Avg_MAPE = avg_mape,
    Median_MAPE = med_mape
  )
}

results_table <- rbind(
  summarize_results(scad_results, "SCAD"),
  summarize_results(mcp_results, "MCP"),
  summarize_results(lasso_glmnet_results, "LASSO_glmnet"),
  summarize_results(regnet_lasso_results, "LAD-LASSO"),
  summarize_results(oracle_results, "ORACLE")
)

print(results_table)


