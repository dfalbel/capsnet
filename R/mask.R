layer_mask <- function(object) {
  keras::create_layer(Mask, object)
}


Mask <- R6::R6Class(
  "Mask",

  inherit = keras::KerasLayer,

  public = list(

    call = function(inputs, mask = NULL) {

      if(class(inputs)[1] == "list") {

        stopifnot(length(inputs) == 2)

        mask <- inputs[[2]]
        inputs <- inputs[[1]]

        mask <- KB$expand_dims(mask, -1L)

      } else {

        x <- inputs
        # Enlarge the range of values in x to make max(new_x)=1 and others < 0
        x <- (x - KB$max(x, 1L, TRUE)) / KB$epsilon() + 1
        mask <- KB$clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

      }

      KB$batch_flatten(inputs*mask)
    },

    compute_output_shape = function(input_shape) {

      if(length(input_shape) == 2) {
        reticulate::tuple(NULL, input_shape[[1]][[2]]*input_shape[[1]][[3]])
      } else {
        reticulate::tuple(NULL, input_shape[[2]]*input_shape[[3]])
      }

    }

  )
)
