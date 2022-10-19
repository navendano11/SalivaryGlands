import segmentation_models_pytorch as pytorch_models



unet3_net1 = pytorch_models.Unet(
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    encoder_depth=3,                # Amount of down- and upsampling of the Unet
    decoder_channels=(256, 128,16),   # Amount of channels
    encoder_weights = 'imagenet' ,         # Model download pretrained weights
    activation = 'sigmoid',            # Activation function to apply after final convolution       
    decoder_use_batchnorm = True
    )

unet3_net2 = pytorch_models.Unet(
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    encoder_depth=3,                # Amount of down- and upsampling of the Unet
    decoder_channels=(256, 128,64),   # Amount of channels
    encoder_weights = 'imagenet' ,         # Model download pretrained weights
    activation = 'sigmoid'            # Activation function to apply after final convolution       
    )