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


#' Apply Conv2D `n_channels` times and concatenate all capsules
#' @param inputs 4D tensor, shape=[None, width, height, channels]
#' @param dim_vector the dim of the output vector of capsule
#' @param n_channels the number of types of capsules
#' @param kernel_size kernel size for convoltions
#' @param strides strides for convolution
#' @param padding padding for convolutions
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






