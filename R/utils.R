#' The non-linear activation used in Capsule. It drives the length of a large vector to
#' near 1 and small vector to 0
#' @param vectors some vectors to be squashed, N-dim tensor
#' @param axis the axis to squash
#' @return a Tensor with same shape as input vectors
squash <- function(vectors, axis = -1L) {
  s_squared_norm <- keras::k_sum(keras::k_square(vectors), axis = axis, keepdims = TRUE)
  scale <- s_squared_norm/(1 + s_squared_norm)/keras::k_sqrt(s_squared_norm + keras::k_epsilon())
  scale * vectors
}

#' Margin Loss
#'
#' @param y_true [None, n_classes]
#' @param y_pred [None, num_capsule]
#'
#' @export
margin_loss <- function(y_true, y_pred) {
  L <- y_true * keras::k_square(keras::k_maximum(0, 0.9 - y_pred)) +
    0.5* (1- y_true) * keras::k_square(keras::k_maximum(0, y_pred - 0.1))

  keras::k_mean(keras::k_sum(L, 1L))
}
