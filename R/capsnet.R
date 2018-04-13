#' Create capsnet model
#'
#' @param input_shape input_shape of images
#' @param n_class number of classes
#' @param num_routing number of routing units
#'
#' @export
create_capsnet <- function(input_shape = c(28, 28, 1), n_class = 10L, num_routing = 3L) {

  x <- keras::layer_input(shape = input_shape, name = "input")

  # Layer 1: Just a conventional Conv2D layer
  conv1 <- keras::layer_conv_2d(
    object = x,
    filters = 256, kernel_size = c(9, 9), strides = c(1,1), padding = "valid",
    activation = "relu", name = "conv1"
  )

  # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
  primarycaps <- primary_cap(
    conv1, dim_vector=8L, n_channels=32L, kernel_size=c(9, 9),
    strides=c(2,2), padding = "valid"
  )

  # Layer 3: Capsule layer. Routing algorithm works here.
  digitcaps <- layer_capsule(
    primarycaps,
    num_capsule = n_class,
    dim_vector = 16L,
    num_routing = num_routing,
    name='digitcaps'
  )

  # Layer 4: This is an auxiliary layer to replace each capsule with its length.
  # Just to match the true label's shape.
  out_caps <- layer_length(digitcaps, name = "capsnet")

  # # Decoder network.
  y <- keras::layer_input(shape=(n_class))
  masked_by_y <- layer_mask(list(digitcaps, y))
  masked <- layer_mask(digitcaps)

  decoder <- keras::keras_model_sequential(name = "decoder")
  decoder %>%
    keras::layer_dense(units = 512, activation = "relu", input_shape = 16*n_class) %>%
    keras::layer_dense(units = 1024, activation = "relu") %>%
    keras::layer_dense(units = prod(input_shape), activation = "sigmoid") %>%
    keras::layer_reshape(target_shape = input_shape, name = "out_recon")


  # Models for training and evaluation (prediction)
  train_model <- keras::keras_model(list(x = x, y = y), list(out_caps, decoder(masked_by_y)))
  eval_model <- keras::keras_model(x, list(out_caps, decoder(masked)))

  list(train_model = train_model, eval_model = eval_model)
}


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

#' Apply Conv2D `n_channels` times and concatenate all capsules
#' @param inputs 4D tensor, shape=[None, width, height, channels]
#' @param dim_vector the dim of the output vector of capsule
#' @param n_channels the number of types of capsules
#' @return  output tensor, shape=[None, num_capsule, dim_vector]
primary_cap <- function(inputs, dim_vector, n_channels, kernel_size, strides, padding) {
  inputs %>%
    keras::layer_conv_2d(
      filters = dim_vector*n_channels,
      kernel_size = kernel_size,
      strides = strides,
      padding = padding,
      name = "primarycap_conv2d"
    ) %>%
    keras::layer_reshape(
      target_shape = c(-1L, dim_vector),
      name = "primarycap_reshape"
    ) %>%
    keras::layer_lambda(
        f = squash,
        name = "primarycap_squash"
      )
}

# define layer wrapper function
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


# define layer wrapper function
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



