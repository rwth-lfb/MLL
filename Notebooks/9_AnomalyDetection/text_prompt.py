"""WinCLIP based text prompt"""
STATE_LEVEL_NORMAL_PROMPTS = (
    lambda c: f'{c}',
    lambda c: f'flawless {c}',
    lambda c: f'perfect {c}',
    lambda c: f'unblemished {c}',
    lambda c: f'{c} without flaw',
    lambda c: f'{c} without defect',
    lambda c: f'{c} without damage',
)

STATE_LEVEL_ABNORMAL_PROMPTS = (
    lambda c: f'damaged {c}',
    lambda c: f'{c} with flaw',
    lambda c: f'{c} with defect',
    lambda c: f'{c} with damage',
)

TEMPLATE_LEVEL_PROMPTS = (
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a photo of the {c} for visual inspection.',
    lambda c: f'a photo of a {c} for visual inspection.',
    lambda c: f'a photo of the {c} for anomaly detection.',
    lambda c: f'a photo of a {c} for anomaly detection.'
)
