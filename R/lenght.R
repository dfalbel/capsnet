layer_length <- function(object, name = NULL, trainable = TRUE) {
  keras::create_layer(Length, object, list(name = name, trainable = trainable))
}

Length <- R6::R6Class(
  "Length",

  inherit = keras::KerasLayer,

  public = list(

    call = function(inputs, mask = NULL) {
      KB$sqrt(KB$sum(KB$square(inputs), -1L))
    },

    compute_output_shape = function(input_shape) {
      reticulate::tuple(input_shape[1:(length(input_shape) - 1)])
    }

  )
)
