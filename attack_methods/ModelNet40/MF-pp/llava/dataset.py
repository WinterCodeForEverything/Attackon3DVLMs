import numpy as np
from torch.utils.data import Dataset
import pickle
from llava.mm_utils import process_images

class ImageNet(Dataset):
    def __init__(self, img_file, image_preocessor, config):
        
        with open(img_file, 'rb') as f:
            self.list_of_images, self.list_of_labels = pickle.load(f) 
        
        self.image_preocessor = image_preocessor
        self.config = config
    
    def __len__(self):
        return len(self.list_of_labels)
    
    def __getitem__(self, idx):
        
        image = self.list_of_images[idx]
        label = self.list_of_labels[idx]
        
        if self.image_preocessor is not None:
                images_tensor = process_images(
                                    [image],
                                    self.image_preocessor,
                                    self.config  
                                 )
                
                image = images_tensor[0]
                
        else:
            image = np.array(image)

        return {
            'images': image,
            'labels': label,
            'indice': idx
        }
    
    

