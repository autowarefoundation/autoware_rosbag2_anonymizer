import torch
from PIL import Image
import open_clip


class OpenClipModel:
    def __init__(self, model_name, pretrained, device) -> None:
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            precision="fp32",
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def __call__(self, image: Image.Image, promt: list[str]) -> torch.Tensor:
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(promt).to(self.device)

        with torch.no_grad(), torch.amp.autocast(self.device):
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            return text_probs.to("cpu")
