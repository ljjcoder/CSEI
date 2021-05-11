from __future__ import absolute_import

from .train_loader import FewShotDataset_train
from .test_loader import FewShotDataset_test
from .train_image_ori_loader import FewShotDataset_train_imgori
from .test_image_ori_loader import FewShotDataset_test_imgori
from .train_inpainting_loader import FewShotDataset_train_inpainting
from .test_inpainting_loader import FewShotDataset_test_inpainting

__loader_factory = {
        'train_loader': FewShotDataset_train,
        'test_loader': FewShotDataset_test,
        'train_imgori_loader': FewShotDataset_train_imgori,
        'test__imgori_loader': FewShotDataset_test_imgori,   
        'train_inpainting_loader': FewShotDataset_train_inpainting,
        'test_inpainting_loader': FewShotDataset_test_inpainting,        
}



def get_names():
    return list(__loader_factory.keys()) 


def init_loader(name, *args, **kwargs):
    if name not in list(__loader_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __loader_factory[name](*args, **kwargs)

