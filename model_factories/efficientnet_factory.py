from efficientnet_pytorch import EfficientNet


def create_model(size):
    return EfficientNet.from_name(f'efficientnet-{size.lower()}')
