#' The non-linear activation used in Capsule. It drives the length of a large vector to
#' near 1 and small vector to 0
#' @param vectors some vectors to be squashed, N-dim tensor
#' @param axis the axis to squash
#' @return a Tensor with same shape as input vectors
squash <- function(vectors, axis = -1L) {
  KB <- keras::backend()
  s_squared_norm <- KB$sum(KB$square(vectors), axis = axis, keepdims = TRUE)
  scale <- s_squared_norm/(1 + s_squared_norm)/KB$sqrt(s_squared_norm + KB$epsilon())
  scale * vectors
}

#' Margin Loss
#'
#' @param y_true [None, n_classes]
#' @param y_pred [None, num_capsule]
#'
#' @export
margin_loss <- function(y_true, y_pred) {
  L <- y_true * KB$square(KB$maximum(0, 0.9 - y_pred)) +
    0.5* (1- y_true) * KB$square(KB$maximum(0, y_pred - 0.1))

  KB$mean(KB$sum(L, 1L))
}
