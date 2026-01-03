from typing import Optional
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms as T
import os

import lightning.pytorch as pl
from PIL import Image
import glob


class MVTecDataModule(pl.LightningDataModule):
    """
    MVTec dataset
    """

    def __init__(
        self,
        data_dir: str = "./Mvtec_AD",
        class_name: str = "wood",
        batch_size: int = 1,
        seed=42,
        resize: int = 224,
        cropsize: int = 224,
        k_shot=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.resize = resize
        self.cropsize = cropsize
        self.data_dir = data_dir
        self.class_name = class_name
        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.batch_size = batch_size
        self.k_shot = k_shot
        self.seed = seed
        self.setup()

    def setup(self, stage: Optional[str] = None):
        train_dataset = MVTec(
            data_dir=self.data_dir,
            class_name=self.class_name,
            is_train=True,
            resize=self.resize,
            cropsize=self.cropsize,
        )

        train_data_length = len(train_dataset)

        if self.k_shot:
            split_list = [self.k_shot, train_data_length - self.k_shot]
        else:
            split_list = [int(train_data_length * 0.8), train_data_length - int(train_data_length * 0.8)]

        self.data_train, self.data_val = random_split(
            dataset=train_dataset, lengths=split_list, generator=torch.Generator().manual_seed(self.seed)
        )

        self.data_test = MVTec(
            data_dir=self.data_dir,
            class_name=self.class_name,
            is_train=False,
            resize=self.resize,
            cropsize=self.cropsize,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_val, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_test, batch_size=self.batch_size, shuffle=False)


class MVTec(Dataset):
    class_names = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    def __init__(
        self,
        data_dir: str = "./Mvtec_AD/",
        class_name: str = "bottle",
        is_train: bool = True,
        is_val: bool = False,
        resize: int = 240,
        cropsize: int = 240,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.resize = resize
        self.is_train = is_train
        self.cropsize = cropsize
        self.is_val = is_val

        # check if the data_dir exists
        if os.path.exists(self.data_dir) is False:
            raise ValueError("data_dir must be a valid path")

        # if class_name not in self.class_names:
        #    raise ValueError("class_name must be one of {}".format(self.class_names))
        # pass
        if not class_name:
            self.class_name = self.class_names
        else:
            self.class_name = [class_name]

        # prepare data
        self.data = self.prepare_data()
        # set transforms
        self.transform_x = T.Compose(
            [
                T.Resize(resize, Image.BICUBIC),
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        self.transform_mask = T.Compose([T.Resize(resize, Image.BICUBIC), T.CenterCrop(cropsize), T.ToTensor()])

    def __getitem__(self, index):
        input_img = self.data["input"][index]
        output = self.data["output"][index]
        mask = self.data["mask"][index]
        category = self.data["category"][index]

        input_img = Image.open(input_img).convert("RGB")
        input_img = self.transform_x(input_img)

        # acquire image from path
        if mask:
            mask = self.transform_mask(Image.open(mask)).to(torch.long)
        else:
            mask = torch.zeros((1, self.cropsize, self.cropsize)).to(torch.long)

        if len(self.class_name) != 1:
            return input_img, output, mask, category
        else:
            return input_img, output, mask

    def __len__(self):
        return len(self.data["input"])

    @staticmethod
    def recursive_search(path, pattern):
        """Recursively search for files with given extension pattern in given path"""
        return glob.glob(os.path.join(path, "**", pattern), recursive=True)

    def prepare_data(self):
        """
        Load data
        dir train contains all good images
        dir_test has broken_large, broken_small, contamination and good types
        """
        # image label placeholder
        x, y, mask, category = [], [], [], []
        for cn_idx, cn in enumerate(self.class_name):
            dir_train = os.path.join(self.data_dir, cn, "train")
            dir_test = os.path.join(self.data_dir, cn, "test")
            if self.is_train:
                if len(self.class_name) != 1:
                    raise AssertionError("Prepare data for multiple category should only be used in evaluation.")
                imgpathlist = glob.glob(os.path.join(dir_train, "**", "*.png"), recursive=True)
                for i in imgpathlist:
                    x.append(i)
                    y.append(0)
                    mask.append(0)
            else:
                # get all the types of test image paths
                test_img_path_list = glob.glob(os.path.join(dir_test, "**", "*.png"))
                for tmp_path in test_img_path_list:
                    # get the type of the image and label
                    split_tmp_path = tmp_path.split(os.sep)
                    anomaly_type, basename = split_tmp_path[-2], split_tmp_path[-1]
                    gt_basename = basename.replace(".png", "_mask.png")
                    x.append(tmp_path)
                    category.append(cn_idx)

                    if anomaly_type == "good":
                        y.append(0)
                        mask.append(0)
                    else:
                        y.append(1)
                        # get the mask image path
                        split_tmp_path[-3], split_tmp_path[-1] = "ground_truth", gt_basename
                        mask.append(os.path.join("/", *split_tmp_path))

        assert len(x) == len(y) == len(mask)
        data = {"input": x, "output": y, "mask": mask, "category": category}
        return data
