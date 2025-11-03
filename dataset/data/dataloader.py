# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.data.modelnet import ModelNet
from dataset.data.objaverse import ObjaverseDataset
# from dataset.data.utils import compute_normals, redirect_normals

def load_dataset(dataset_type, data_path, category_file=None, anno_path=None, 
                 do_preprocess=True, normalized_pc=True,
                 has_color=True, has_normal=True, redirect_normals=False,
                 use_color=True, use_normal=True, pts_processor=None, text_processor=None):
    # print(f"Loading {split} split of ModelNet datasets.")
    if dataset_type == 'ModelNet':
        dataset = ModelNet(data_path=data_path, category_file=category_file,
                           do_preprocess=do_preprocess, normalized_pc=normalized_pc,
                           has_color=has_color, has_normal=has_normal,
                           redirect_normals=redirect_normals, use_color=use_color, use_normal=use_normal,
                           pts_processor=pts_processor, text_processor=text_processor )
    elif dataset_type == 'Objaverse':
        dataset = ObjaverseDataset(data_path=data_path, anno_path=anno_path, 
                                   normalized_pc=normalized_pc, has_color=has_color, has_normal=has_normal,
                                   redirect_normals=redirect_normals, use_color=use_color, use_normal=use_normal,
                                   pts_processor=pts_processor, text_processor=text_processor)
    print("Done!")
    return dataset


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
