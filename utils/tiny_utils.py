import subprocess
import torch

def set_info_logger():
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

def find_free_gpu(logger):
    if torch.cuda.is_available():
        smi_output = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        free_memory = [int(x) for x in smi_output.stdout.strip().split('\n')]
        most_free_gpu = free_memory.index(max(free_memory))
        device = torch.device(f"cuda:{most_free_gpu}")
        logger.info(f"Using GPU: {most_free_gpu}, free memory: {int(free_memory[most_free_gpu]/1024)} GiB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device