layer_capsule <- function(object, num_capsule, dim_vector, num_routing = 3L,
                          kernel_initializer = "glorot_uniform",
                          bias_initializer = "zeros",
                          name = NULL,
                          trainable = TRUE) {
  keras::create_layer(CapsuleLayer, object, list(
    as.integer(num_capsule), as.integer(dim_vector), num_routing = as.integer(num_routing),
    kernel_initializer = "glorot_uniform",
    bias_initializer = "zeros"
  ))
}


CapsuleLayer <- R6::R6Class(
  "CapsuleLayer",

  inherit = keras::KerasLayer,

  public = list(

    num_capsule = NULL,
    dim_vector = NULL,
    num_routing = NULL,
    kernel_initializer = NULL,
    bias_initializer = NULL,

    input_num_capsule = NULL,
    input_dim_vector = NULL,

    W = NULL,
    bias  = NULL,

    built = NULL,

    initialize = function(num_capsule, dim_vector, num_routing = 3L,
                          kernel_initializer = "glorot_uniform",
                          bias_initializer = "zeros") {


      self$num_capsule <- as.integer(num_capsule)
      self$dim_vector <- as.integer(dim_vector)
      self$num_routing <- as.integer(num_routing)
      self$kernel_initializer <- keras_caps$initializers$get(kernel_initializer)
      self$bias_initializer <- keras_caps$initializers$get(bias_initializer)
    },

    build = function(input_shape) {

      stopifnot(length(input_shape) >= 3)

      self$input_num_capsule <- as.integer(input_shape[2])
      self$input_dim_vector <- as.integer(input_shape[3])

      # Transform matrix
      self$W <- self$add_weight(
        shape=c(self$input_num_capsule, self$num_capsule, self$input_dim_vector, self$dim_vector),
        initializer=self$kernel_initializer,
        name='W'
      )

      # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
      self$bias <- self$add_weight(
        shape=c(1, self$input_num_capsule, self$num_capsule, 1, 1),
        initializer = self$bias_initializer,
        name = 'bias',
        trainable = FALSE
      )

      self$built <- TRUE
    },

    call = function(inputs, training=NULL) {

      # inputs.shape=[None, input_num_capsule, input_dim_vector]
      # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
      inputs_expand <- KB$expand_dims(KB$expand_dims(inputs, 2L), 2L)

      # Replicate num_capsule dimension to prepare being multiplied by W
      # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
      inputs_tiled <- KB$tile(inputs_expand, c(1L, 1L, self$num_capsule, 1L, 1L))

      # Compute `inputs * W` by scanning inputs_tiled on dimension 0. This is faster but requires Tensorflow.
      # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
      inputs_hat <- tensorflow::tf$scan(
        function(ac, x) {
          KB$batch_dot(x, self$W, c(3L, 2L))
        },
        elems = inputs_tiled,
        initializer = KB$zeros(c(self$input_num_capsule, self$num_capsule, 1, self$dim_vector))
      )

      stopifnot(self$num_routing > 0)

      # Routing algorithm. Use iteration.
      for(i in seq_along(self$num_routing)) {
        sigm <- tensorflow::tf$nn$softmax(self$bias, dim = 2L)
        outputs <- squash(KB$sum(sigm * inputs_hat, 1L, keepdims = TRUE))

        if(i == self$num_routing) {
          self$bias = self$bias + KB$sum(inputs_hat * outputs, -1L, keepdims = TRUE)
        }
      }

      KB$reshape(outputs, c(-1L, self$num_capsule, self$dim_vector))

    },

    compute_output_shape = function(input_shape) {
      reticulate::tuple(NULL, self$num_capsule, self$dim_vector)
    }
  )
)
