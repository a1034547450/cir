import os

import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal

import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset



class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield :a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions'] when
             split in ['train', 'val']
            - ['reference_image', 'reference_name', 'relative_captions'] when split == test
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'], dress_types: List[str],
                 mode: Literal['relative', 'classic'], no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the FashionIQ dataset
        :param split: dataset split, should be in ['train, 'val', 'test']
        :param dress_types: list of fashionIQ categories, each category should be in ['dress', 'shirt', 'toptee']
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
            - In 'relative' mode the dataset yield dict with keys:
                - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions']
                 when split in ['train', 'val']
                - ['reference_image', 'reference_name', 'relative_captions'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.no_duplicates = no_duplicates

        # Validate the inputs
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(dataset_path / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # Remove duplicats from
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['candidate'] not in seen:
                    seen.add(triplet['candidate'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(dataset_path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> Dict:
        try:
            if self.mode == 'relative':
                relative_captions = self.triplets[index]['captions'][0].strip('.?, ')+' and ' +self.triplets[index]['captions'][1].strip('.?, ')
                reference_name = self.triplets[index]['candidate']

                if self.split in ['train', 'val']:
                    reference_image_path = str(self.dataset_path / 'images' / f"{reference_name}.jpg")
                    # reference_image = PIL.Image.open(reference_image_path)
                    target_name = self.triplets[index]['target']
                    target_image_path = str(self.dataset_path / 'images' / f"{target_name}.jpg")
                    # target_image = PIL.Image.open(target_image_path)

                    return {
                        'reference_image_path': reference_image_path,
                        'reference_name': reference_name,
                        'target_image_path': target_image_path,
                        'target_name': target_name,
                        'relative_caption': relative_captions
                    }

                elif self.split == 'test':
                    reference_image_path = self.dataset_path / 'images' / f"{reference_name}.jpg"
                    reference_image = PIL.Image.open(reference_image_path)
                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'relative_captions': relative_captions
                    }

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = str(self.dataset_path / 'images' / f"{image_name}.jpg")
                # image = PIL.Image.open(image_path)
                # print({
                #     'image_path': image_path,
                #     'image_name': image_name
                # })
                return {
                    'image_path': image_path,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
