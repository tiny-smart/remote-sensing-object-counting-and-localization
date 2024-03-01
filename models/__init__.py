from .backbones import *
from .transformer import *
from .pet import build_pet

def build_model(args,test=False):
    return build_pet(args,test)

