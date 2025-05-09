import torch
import os
from PIL import Image
import json

# import orjson
from torchvision import transforms
import pytorch_lightning as pl

from layout_diffusion.dataset.util import image_normalize

import torch.nn.functional as F
import glob
import random

import zipfile

# import sys

# sys.path.append('../webuidata')

# import download_partial_data_webui

DEVICE_SCALE = {
    "default": 1,
    "iPad-Mini": 2,
    "iPad-Pro": 2,
    "iPhone-13 Pro": 3,
    "iPhone-SE": 3,
}


def makeMultiHotVec(idxs, num_classes):
    vec = [1 if i in idxs else 0 for i in range(num_classes)]
    return vec


# todo, maybe add more image transformations for data augmentation
class WebUIPilotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        root_dir="../../",
        class_map_file="class_map.json",
        min_area=10,
        device_scale=DEVICE_SCALE,
    ):
        super(WebUIPilotDataset, self).__init__()
        self.keys = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".json")
        ]
        # TODO: remove this (only for debugging)
        if data_dir == "val_dataset":
            self.keys = self.keys[:1000]
        self.root_dir = root_dir
        self.min_area = min_area
        self.device_scale = device_scale
        with open(class_map_file, "r") as f:
            class_map = json.load(f)

        self.idx2Label = class_map["idx2Label"]
        self.label2Idx = class_map["label2Idx"]
        self.num_classes = max([int(k) for k in self.idx2Label.keys()]) + 1
        self.img_transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        try:
            idx = idx % len(self.keys)
            key = self.keys[idx]
            with open(key, "r") as f:
                key_dict = json.load(f)
            url_path = os.path.join(
                self.root_dir, key_dict["key_name"]
            )  # path to url.txt file
            url_path = url_path.replace("\\", "/")
            key_filename = url_path.split("/")[-1]
            device_name = "-".join(key_filename.split("-")[:-1])
            img_path = url_path.replace("-url.txt", "-screenshot.png")
            img_pil = Image.open(img_path).convert("RGB")
            img = self.img_transforms(img_pil)
            target = {}
            boxes = []
            labels = []
            for i in range(len(key_dict["labels"])):
                box = key_dict["contentBoxes"][i]
                # skip invalid boxes
                if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                    continue
                if box[3] <= box[1] or box[2] <= box[0]:
                    continue
                if (box[3] - box[1]) * (
                    box[2] - box[0]
                ) <= self.min_area:  # get rid of really small elements
                    continue
                boxes.append(box)
                label = key_dict["labels"][i]
                labelIdx = [
                    (
                        self.label2Idx[label[li]]
                        if label[li] in self.label2Idx
                        else self.label2Idx["OTHER"]
                    )
                    for li in range(len(label))
                ]
                labelHot = makeMultiHotVec(set(labelIdx), self.num_classes)
                labels.append(labelHot)

            if len(boxes) > 200:
                # print("skipped due to too many objects", len(boxes))
                return self.__getitem__(idx + 1)

            boxes = torch.tensor(boxes, dtype=torch.float)
            boxes *= self.device_scale[device_name]

            labels = torch.tensor(labels, dtype=torch.long)

            target["boxes"] = boxes if len(boxes.shape) == 2 else torch.zeros(0, 4)
            target["labels"] = labels
            target["image_id"] = torch.tensor([idx])
            target["area"] = (
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                if len(boxes.shape) == 2
                else torch.zeros(0)
            )
            target["iscrowd"] = (
                torch.zeros((boxes.shape[0],), dtype=torch.long)
                if len(boxes.shape) == 2
                else torch.zeros(0, dtype=torch.long)
            )

            return img, target  # return image and target dict
        except Exception as e:
            print("failed", idx, str(e))
            return self.__getitem__(idx + 1)


class WebUIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_file,
        boxes_dir="../../downloads/webui-boxes/all_data",
        rawdata_screenshots_dir="../../downloads/ds",
        class_map_file="class_map.json",
        min_area=100,
        device_scale=DEVICE_SCALE,
        max_boxes=100,
        max_skip_boxes=100,
        image_size=(128, 128),
        layout_length=10,
        **kwargs
    ):
        super(WebUIDataset, self).__init__()
        self.max_boxes = max_boxes
        self.max_skip_boxes = max_skip_boxes
        self.keys = []

        with open(split_file, "r") as f:
            boxes_split = json.load(f)

        rawdata_directory = rawdata_screenshots_dir
        for folder in [f for f in os.listdir(boxes_dir) if f in boxes_split]:
            for file in os.listdir(os.path.join(boxes_dir, folder)):
                if os.path.exists(
                    os.path.join(
                        rawdata_directory,
                        folder,
                        file.replace(".json", "-screenshot.webp"),
                    )
                ):
                    self.keys.append(os.path.join(boxes_dir, folder, file))

        self.min_area = min_area
        self.device_scale = device_scale
        with open(class_map_file, "r") as f:
            class_map = json.load(f)
        self.computed_boxes_directory = boxes_dir
        self.rawdata_directory = rawdata_directory
        self.idx2Label = class_map["idx2Label"]
        self.label2Idx = class_map["label2Idx"]
        self.num_classes = max([int(k) for k in self.idx2Label.keys()]) + 1
        self.img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=True),
            ]
        )
        # image_normalize()])

        self.image_size = (image_size, image_size)
        self.layout_length = layout_length

    def __len__(self):
        return len(self.keys)

    def total_objects(self):
        to = 0
        for i in range(len(self.keys)):
            with open(self.keys[i], "r") as f:
                key_dict = json.load(f)
            to += len(key_dict["labels"])
        return to

    def __getitem__(self, idx):
        # try:
        idx = idx % len(self.keys)
        key = self.keys[idx]
        with open(key, "r") as f:
            key_dict = json.load(f)

        img_path = key.replace(".json", "-screenshot.webp")
        img_path = img_path.replace(
            self.computed_boxes_directory, self.rawdata_directory
        )

        key_filename = img_path.split("/")[-1]
        device_name = "-".join(key_filename.split("-")[:-1])

        img_pil = Image.open(img_path).convert("RGB")
        org_size = img_pil.size
        img = self.img_transforms(img_pil)
        target = {}
        boxes = []
        labels = []
        labelNames = []
        scale = self.device_scale[device_name.split("_")[0]]

        inds = list(range(len(key_dict["labels"])))
        random.shuffle(inds)

        for i in inds:
            box = key_dict["contentBoxes"][i]
            box[0] *= scale # w
            box[1] *= scale # h
            box[2] *= scale # w
            box[3] *= scale # h

            box[0] = round(min(max(0, box[0]), org_size[0]) / (org_size[0] / self.image_size[0]))
            box[1] = round(min(max(0, box[1]), org_size[1]) / (org_size[1] / self.image_size[1]))
            box[2] = round(min(max(0, box[2]), org_size[0]) / (org_size[0] / self.image_size[0]))
            box[3] = round(min(max(0, box[3]), org_size[1]) / (org_size[1] / self.image_size[1]))

            # skip invalid boxes
            if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                continue
            if box[3] <= box[1] or box[2] <= box[0]:
                continue
            if (box[3] - box[1]) * (
                box[2] - box[0]
            ) <= self.min_area:  # get rid of really small elements
                continue
            boxes.append(box)
            label = key_dict["labels"][i]
            labelIdx = [
                (
                    self.label2Idx[label[li]]
                    if label[li] in self.label2Idx
                    else self.label2Idx["OTHER"]
                )
                for li in range(len(label))
            ]
            labelHot = makeMultiHotVec(set(labelIdx), self.num_classes)
            labelNames.append(label)
            labels.append(labelHot)

        if len(boxes) > self.max_skip_boxes:
            # print("skipped due to too many objects", len(boxes))
            return self.__getitem__(idx + 1)

        boxes = torch.tensor(boxes, dtype=torch.float)

        labels = torch.tensor(labels, dtype=torch.float)

        target["obj_bbox"] = boxes if len(boxes.shape) == 2 else torch.zeros(0, 4)
        target["obj_class"] = labels if len(labels.shape) == 2 else torch.zeros(0, self.num_classes)
        target["obj_class_name"] = labelNames
        target["image_id"] = torch.tensor([idx])
        target["area"] = (
            (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if len(boxes.shape) == 2
            else torch.zeros(0)
        )
        target["iscrowd"] = (
            torch.zeros((boxes.shape[0],), dtype=torch.long)
            if len(boxes.shape) == 2
            else torch.zeros(0, dtype=torch.long)
        )
        target["is_valid_obj"] = torch.ones(len(target["obj_bbox"]))
        target["num_obj"] = len(inds)

        for k in target:
            if not isinstance(target[k], int):
                target[k] = target[k][: self.max_boxes]

        target["obj_bbox"] = torch.nn.functional.pad(
            target["obj_bbox"],
            (0, 0, 0, self.layout_length - len(target["obj_bbox"])),
            mode="constant",
            value=0,
        )
        target["obj_class"] = torch.nn.functional.pad(
            target["obj_class"],
            (0, 0, 0, self.layout_length - len(target["obj_class"])),
            mode="constant",
            value=0,
        )
        target["is_valid_obj"] = torch.nn.functional.pad(
            target["is_valid_obj"],
            (0, self.layout_length - len(target["is_valid_obj"])),
            mode="constant",
            value=0,
        )

        return img, target  # return image and target dict

class WebUIDatasetTest(torch.utils.data.Dataset):
    def __init__(
        self,
        boxes_file,
        class_map_file="class_map.json",
        min_area=100,
        device_scale=DEVICE_SCALE,
        max_boxes=100,
        max_skip_boxes=100,
        resized_size=128,
        layout_length=10,
        base_size=(1920,1080),
        **kwargs
    ):
        super(WebUIDatasetTest, self).__init__()
        self.max_boxes = max_boxes
        self.max_skip_boxes = max_skip_boxes
        self.keys = []

        with open(boxes_file, "r") as f:
            self.boxes = json.load(f)

        self.min_area = min_area
        self.device_scale = device_scale
        with open(class_map_file, "r") as f:
            class_map = json.load(f)
        self.idx2Label = class_map["idx2Label"]
        self.label2Idx = class_map["label2Idx"]
        self.num_classes = max([int(k) for k in self.idx2Label.keys()]) + 1
        # image_normalize()])

        self.image_size = (resized_size, resized_size)
        self.layout_length = layout_length
        self.base_size = base_size

    def __len__(self):
        return len(self.boxes)

    def total_objects(self):
        to = 0
        for i in range(len(self.keys)):
            with open(self.keys[i], "r") as f:
                key_dict = json.load(f)
            to += len(key_dict["labels"])
        return to

    def __getitem__(self, idx):
        # try:
        idx = idx % len(self.boxes)

        target = {}
        boxes = []
        masks = []
        labels = []
        labelNames = []

        key_dict = self.boxes[idx]

        for i in range(len(key_dict["contentBoxes"])):
            box = key_dict["contentBoxes"][i]

            box[0] = round(min(max(0, box[0]), self.base_size[0]) / (self.base_size[0] / self.image_size[0]))
            box[1] = round(min(max(0, box[1]), self.base_size[1]) / (self.base_size[1] / self.image_size[1]))
            box[2] = round(min(max(0, box[2]), self.base_size[0]) / (self.base_size[0] / self.image_size[0]))
            box[3] = round(min(max(0, box[3]), self.base_size[1]) / (self.base_size[1] / self.image_size[1]))

            # box[0] *= self.image_size[0]
            # box[1] *= self.image_size[1]
            # box[2] *= self.image_size[0]
            # box[3] *= self.image_size[1]

            # skip invalid boxes
            if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                continue
            if box[3] <= box[1] or box[2] <= box[0]:
                continue
            if (box[3] - box[1]) * (
                box[2] - box[0]
            ) <= self.min_area:  # get rid of really small elements
                continue
            boxes.append(box)
            label = key_dict["labels"][i]
            labelIdx = [
                (
                    self.label2Idx[label[li]]
                    if label[li] in self.label2Idx
                    else self.label2Idx["OTHER"]
                )
                for li in range(len(label))
            ]
            labelHot = makeMultiHotVec(set(labelIdx), self.num_classes)
            labelNames.append(label)
            labels.append(labelHot)

        if len(boxes) > self.max_skip_boxes:
            # print("skipped due to too many objects", len(boxes))
            return self.__getitem__(idx + 1)

        boxes = torch.tensor(boxes, dtype=torch.float)

        labels = torch.tensor(labels, dtype=torch.long)

        target["obj_bbox"] = boxes if len(boxes.shape) == 2 else torch.zeros(0, 4)
        target["obj_class"] = labels
        target["obj_class_name"] = labelNames
        target["image_id"] = torch.tensor([idx])
        target["area"] = (
            (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if len(boxes.shape) == 2
            else torch.zeros(0)
        )
        target["iscrowd"] = (
            torch.zeros((boxes.shape[0],), dtype=torch.long)
            if len(boxes.shape) == 2
            else torch.zeros(0, dtype=torch.long)
        )
        target["is_valid_obj"] = torch.ones(len(target["obj_bbox"]))
        target["num_obj"] = len(key_dict["contentBoxes"])

        for k in target:
            if not isinstance(target[k], int):
                target[k] = target[k][: self.max_boxes]

        target["obj_bbox"] = torch.nn.functional.pad(
            target["obj_bbox"],
            (0, 0, 0, self.layout_length - len(target["obj_bbox"])),
            mode="constant",
            value=0,
        )
        target["obj_class"] = torch.nn.functional.pad(
            target["obj_class"],
            (0, 0, 0, self.layout_length - len(target["obj_class"])),
            mode="constant",
            value=0,
        )
        target["is_valid_obj"] = torch.nn.functional.pad(
            target["is_valid_obj"],
            (0, self.layout_length - len(target["is_valid_obj"])),
            mode="constant",
            value=0,
        )

        return torch.zeros((3, *self.image_size)), target  # return image and target dict

    # except Exception as e:
    #     print("failed", idx, str(e))
    #     return self.__getitem__(idx + 1)


# https://github.com/pytorch/vision/blob/5985504cc32011fbd4312600b4492d8ae0dd13b4/references/detection/utils.py#L203
def wui_collate_fn_for_layout(batch):
    return tuple(zip(*batch))


class WebUIPilotDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, num_workers=2):
        super(WebUIPilotDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = WebUIPilotDataset(data_dir="train_dataset")
        self.val_dataset = WebUIPilotDataset(data_dir="val_dataset")
        self.test_dataset = WebUIPilotDataset(data_dir="test_dataset")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )


class WebUIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_split_file,
        val_split_file="../../downloads/val_split_webui.json",
        test_split_file="../../downloads/test_split_webui.json",
        batch_size=8,
        num_workers=4,
    ):
        super(WebUIDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = WebUIDataset(split_file=train_split_file)
        self.val_dataset = WebUIDataset(split_file=val_split_file)
        self.test_dataset = WebUIDataset(split_file=test_split_file)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )  # shuffle so that we can eval on subset

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )


def build_wui_dsets(cfg, mode="train"):
    assert mode in ["train", "val", "test"]
    params = cfg.data.parameters
    dataset = WebUIDataset(**params)

    num_objs = dataset.total_objects()
    num_imgs = len(dataset)
    print("%s dataset has %d images and %d objects" % (mode, num_imgs, num_objs))
    print("(%.2f objects per image)" % (float(num_objs) / num_imgs))

    return dataset

def build_wui_dsets_test(cfg, mode="train"):
    assert mode in ["train", "val", "test"]
    params = cfg.data.parameters
    dataset = WebUIDatasetTest(**params)

    num_objs = dataset.total_objects()
    num_imgs = len(dataset)
    print("%s dataset has %d images and %d objects" % (mode, num_imgs, num_objs))
    print("(%.2f objects per image)" % (float(num_objs) / num_imgs))

    return dataset