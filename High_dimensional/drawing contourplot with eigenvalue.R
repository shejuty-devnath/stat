install.packages("pracma")
library(pracma) 

X <- c(3, 3)

S1 <- matrix(c(6, 3, 3, 6), nrow=2, byrow=TRUE)
S2 <- matrix(c(6, -3, -3, 6), nrow=2, byrow=TRUE)

eigen_S1 <- eigen(S1)

eigen_S1$values   # Eigenvalues
eigen_S1$vectors  # Eigenvectors

eigen_S2 <- eigen(S2)

eigen_S2$values   # Eigenvalues
eigen_S2$vectors  # Eigenvectors


x_vals <- seq(-5, 11, length.out=200)
y_vals <- seq(-5, 11, length.out=200)

grid <- meshgrid(x_vals, y_vals)
X_grid <- grid$X
Y_grid <- grid$Y

compute_quad_form <- function(X_grid, Y_grid, X, S) {
  V1 <- X_grid - X[1]
  V2 <- Y_grid - X[2]
  
  f_values <- V1 * (S[1,1] * V1 + S[1,2] * V2) +
    V2 * (S[2,1] * V1 + S[2,2] * V2)
  return(f_values)
}


f_values_S1 <- compute_quad_form(X_grid, Y_grid, X, S1)

f_values_S2 <- compute_quad_form(X_grid, Y_grid, X, S2)


#############

contour(x_vals, y_vals, f_values_S1, levels=pretty(range(f_values_S1), 15),
        xlab="x", ylab="y", main="Contour Plot for S1 and S2")
points(X[1], X[2], col="red", pch=19)  

contour(x_vals, y_vals, f_values_S2, levels=pretty(range(f_values_S2), 15),
        col = "blue", add = TRUE)  
points(X[1], X[2], col="red", pch=19)  
#############
