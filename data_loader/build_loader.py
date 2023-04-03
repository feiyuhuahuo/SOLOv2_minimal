#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import random
import torch
import bisect
import torch.utils.data as data
import numpy as np
import itertools
import pdb


def val_collate(batch):
    imgs = torch.tensor(batch[0][0].transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return imgs, batch[0][1], batch[0][2], batch[0][3]


def detect_collate(batch):
    imgs = torch.tensor(batch[0][0].transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return imgs, batch[0][1], batch[0][2], batch[0][3]


class group_sampler:
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven
        self.groups = torch.unique(self.group_ids).sort(0)[0]
        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was not sampled, and a non-negative number
        # indicating the order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5, the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0
        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but they are grouped by clusters.
        # Find the permutation between different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the ordering as coming from
        # the first element of each batch, and sort correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor([inv_sampled_ids_map[s] for s in first_element_of_batch])

        # permute the batches so that they approximately follow the order from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches

        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True

        return len(self._batches)


def complement_batch(batch):
    valid_batch = [aa for aa in batch if aa[0] is not None]
    lack_len = len(batch) - len(valid_batch)
    if lack_len > 0:
        for i in range(lack_len):
            valid_batch.append(valid_batch[i])

    return valid_batch


def batch_collator_random_pad(batch):
    batch = complement_batch(batch)
    img_batch, labels_batch, bboxes_batch, masks_batch = list(zip(*batch))
    max_size = tuple(max(s) for s in zip(*[img.shape for img in img_batch]))
    assert max_size[0] % 32 == 0 and max_size[1] % 32 == 0, 'shape error in batch_collator'

    batch_shape = (len(img_batch), max_size[0], max_size[1], max_size[2])
    pad_imgs_batch = np.zeros(batch_shape, dtype=img_batch[0].dtype)

    pad_labels_batch, pad_bboxes_batch, pad_masks_batch = [], [], []
    for i, img in enumerate(img_batch):
        ori_h, ori_w = img.shape[:2]
        new_h = random.randint(0, (max_size[0] - ori_h))
        new_w = random.randint(0, (max_size[1] - ori_w))

        pad_imgs_batch[i, new_h:new_h + ori_h, new_w:new_w + ori_w, :] = img

        pad_labels_batch.append(torch.tensor(labels_batch[i], dtype=torch.int64))

        ori_bboxes = bboxes_batch[i]
        pad_bboxes = ori_bboxes.copy()
        pad_bboxes[:, [0, 2]] = pad_bboxes[:, [0, 2]] + new_w
        pad_bboxes[:, [1, 3]] = pad_bboxes[:, [1, 3]] + new_h
        pad_bboxes_batch.append(torch.tensor(pad_bboxes, dtype=torch.float32))

        ori_masks = masks_batch[i]
        pad_masks = np.zeros((batch_shape[1], batch_shape[2], ori_masks.shape[2]), dtype='uint8')
        pad_masks[new_h:new_h + ori_h, new_w:new_w + ori_w, :] = ori_masks
        pad_masks_batch.append(pad_masks.transpose(2, 0, 1).astype('uint8'))

    pad_imgs_batch = pad_imgs_batch.transpose(0, 3, 1, 2)
    return torch.tensor(pad_imgs_batch, dtype=torch.float32), pad_labels_batch, pad_bboxes_batch, pad_masks_batch


def batch_collator_tl_pad(batch):
    batch = complement_batch(batch)
    img_batch, labels_batch, bboxes_batch, masks_batch = list(zip(*batch))
    max_size = tuple(max(s) for s in zip(*[img.shape for img in img_batch]))
    assert max_size[0] % 32 == 0 and max_size[1] % 32 == 0, 'shape error in batch_collator'

    batch_shape = (len(img_batch), max_size[0], max_size[1], max_size[2])
    pad_imgs_batch = np.zeros(batch_shape, dtype=img_batch[0].dtype)

    labels_list, bboxes_list, masks_list = [], [], []
    for i, img in enumerate(img_batch):
        ori_h, ori_w = img.shape[:2]

        pad_imgs_batch[i, 0:ori_h, 0:ori_w, :] = img

        labels_list.append(torch.tensor(labels_batch[i], dtype=torch.int64))
        bboxes_list.append(torch.tensor(bboxes_batch[i], dtype=torch.float32))
        masks_list.append(masks_batch[i].transpose(2, 0, 1).astype('uint8'))

    pad_imgs_batch = pad_imgs_batch.transpose(0, 3, 1, 2)
    return torch.tensor(pad_imgs_batch, dtype=torch.float32), labels_list, bboxes_list, masks_list


def make_data_loader(cfg):
    dataset = cfg.dataset(cfg)

    if cfg.mode == 'train':
        aspect_ratios = dataset.get_aspect_ratios()
        # group in two cases: those with width / height > 1, and the other way around
        group_ids = list(map(lambda y: bisect.bisect_right([1], y), aspect_ratios))
        sampler = data.RandomSampler(dataset)
        batch_sampler = group_sampler(sampler, group_ids, cfg.train_bs, drop_uneven=False)  # same as drop_last
        return data.DataLoader(dataset, num_workers=cfg.train_workers,
                               batch_sampler=batch_sampler, collate_fn=batch_collator_random_pad)
    else:
        if cfg.mode == 'val':
            collator = val_collate
        elif cfg.mode == 'detect':
            collator = detect_collate
        return data.DataLoader(dataset, num_workers=2, batch_size=cfg.val_bs, shuffle=False, collate_fn=collator)
