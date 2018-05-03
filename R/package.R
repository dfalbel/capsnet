# Main Keras module
keras_caps <- NULL

.onLoad <- function(libname, pkgname) {
  keras_caps <<- keras::implementation()
}

#' Pipe operator
#'
#' See \code{\link[magrittr]{\%>\%}} for more details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @import magrittr
#' @usage lhs \%>\% rhs
NULL

