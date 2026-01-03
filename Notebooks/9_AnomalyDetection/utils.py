import open_clip
from open_clip.model import CLIP

def get_clip_encoders(
    backbone: str = 'ViT-B-16',
    pretrained: str = 'laion400m_e32',
    only_model: bool = False,
):
    """
    extracted the pretrained clip model, tokenizer and hyperparameters setting
    """
    model: CLIP
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained)
    except Exception as e:
        print(f"Error {e}, select the correct pretrained model")

    # change clip image encoder hyperparameters for few-shot learning
    model = model.eval()

    tokenizer = open_clip.get_tokenizer(model_name=backbone)

    logit_scale = model.logit_scale
    clip_config = open_clip.get_model_config(backbone)


    if only_model:
        output = model.eval()
    else:
        output = {
            "clip_model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "logit_scale": logit_scale,
            "clip_config": clip_config
        }
    return output
