# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import nvsmi
import torch

def nvidia_smi():
    # Use a breakpoint in the code line below to debug your script.
    for g in nvsmi.get_available_gpus():
        print("gpu: [%s] %s utilization: %s" % (g.id, g.name, g.gpu_util))


def get_free_gpu():
    if not torch.cuda.is_available():
        return None

    for g in nvsmi.get_available_gpus():
        if g.gpu_util/1.0 == 0.0:
            cuda = "cuda:{}".format(g.id)
            print("[%s] [%s] [%s] is free , return [%s]" % (g.id, g.name, g.uuid, cuda))
            return cuda
    return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nvidia_smi()
