import timm
import torch
import numpy as np


# place model
model = timm.create_model('mobilevit_xxs', num_classes=257)

# specifications
gpu_or_cpu = input('GPU_or_CPU? ')
image_size = 224

print(f'results for {gpu_or_cpu} on an image {(3, image_size, image_size)} ...')
if gpu_or_cpu=='GPU':
    GPU = torch.device("cuda:0")
    model.to(GPU)
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float).to(GPU)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)

    metrics = []
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            torch.cuda.empty_cache()
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            metrics.append(curr_time)
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    # inference time
    print(f'inference time on GPU 3090: {mean_syn} msec')

if gpu_or_cpu=='CPU':
    CPU = torch.device("cpu")
    model.to(CPU)
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float).to(CPU)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)

    metrics = []
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            torch.cuda.empty_cache()
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            metrics.append(curr_time)
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    # inference time
    print(f'inference time on CPU: {mean_syn} msec')
