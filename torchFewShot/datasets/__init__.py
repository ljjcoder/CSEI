from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .miniImageNet import miniImageNet



__imgfewshot_factory = {
        'miniImageNet': miniImageNet,
}


def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    #print('name is: ',name,'kwargs: ',**kwargs)
    #exit(0)
    return __imgfewshot_factory[name](**kwargs)

