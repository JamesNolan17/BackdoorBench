import subprocess
#import torch

def set_info_logger():
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

def find_free_gpu(logger):
    blacklist=[3]
    smi_output = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
    free_memory = [int(x) for x in smi_output.stdout.strip().split('\n')]
    
    # Filter out blacklisted GPUs
    available_gpus = [(idx, mem) for idx, mem in enumerate(free_memory) if idx not in blacklist]
    
    if available_gpus:
        most_free_gpu = max(available_gpus, key=lambda x: x[1])[0]
        logger.info(f"Using GPU: {most_free_gpu}, free memory: {int(free_memory[most_free_gpu]/1024)} GiB")
    else:
        logger.error("No available GPUs found that are not blacklisted")
        return None
    
    return most_free_gpu