## fig_shrinkage_comparison.R
## Comparison of ridge multiplicative shrinkage vs lasso soft-thresholding

z <- seq(-5, 5, length.out = 501)
lambdas <- c(0.5, 1, 2)
cols <- c("#0072B2", "#D55E00", "#009E73")  # colour-blind safe
ltys <- c(1, 1, 1)

# Ridge: beta_hat = z / (1 + lambda)
ridge <- function(z, lam) z / (1 + lam)

# Lasso: beta_hat = sign(z) * max(|z| - lambda, 0)
lasso <- function(z, lam) sign(z) * pmax(abs(z) - lam, 0)

pdf("figs/shrinkage_comparison.pdf", width = 12, height = 5.5, pointsize = 10)
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 2.5, 1), cex.lab = 1.15, cex.main = 1.2)

# --- Left panel: Ridge ---
plot(NULL, xlim = c(-5, 5), ylim = c(-5, 5),
     xlab = "OLS estimate z", ylab = expression("Penalised estimate " * hat(beta)),
     main = "Ridge: multiplicative shrinkage")
abline(0, 1, lty = 2, col = "grey50", lwd = 1.5)  # 45-degree reference
abline(h = 0, col = "grey80"); abline(v = 0, col = "grey80")
for (i in seq_along(lambdas)) {
  lines(z, ridge(z, lambdas[i]), col = cols[i], lwd = 2.2)
}
legend("topleft", legend = c(
  expression(lambda == 0.5),
  expression(lambda == 1),
  expression(lambda == 2),
  "No penalty"
), col = c(cols, "grey50"), lwd = c(2.2, 2.2, 2.2, 1.5),
lty = c(1, 1, 1, 2), bty = "n", cex = 0.9)

# --- Right panel: Lasso ---
plot(NULL, xlim = c(-5, 5), ylim = c(-5, 5),
     xlab = "OLS estimate z", ylab = expression("Penalised estimate " * hat(beta)),
     main = "Lasso: soft-thresholding")
abline(0, 1, lty = 2, col = "grey50", lwd = 1.5)
abline(h = 0, col = "grey80"); abline(v = 0, col = "grey80")
for (i in seq_along(lambdas)) {
  lines(z, lasso(z, lambdas[i]), col = cols[i], lwd = 2.2)
}
legend("topleft", legend = c(
  expression(lambda == 0.5),
  expression(lambda == 1),
  expression(lambda == 2),
  "No penalty"
), col = c(cols, "grey50"), lwd = c(2.2, 2.2, 2.2, 1.5),
lty = c(1, 1, 1, 2), bty = "n", cex = 0.9)

dev.off()
cat("Saved figs/shrinkage_comparison.pdf\n")
