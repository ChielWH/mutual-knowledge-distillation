from efficientnet_pytorch import EfficientNet


def EfficientNetB(size=6):
    assert size in range(8)
    return EfficientNet.from_name(f'efficientnet-b{size}')
