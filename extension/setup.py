"""Build gfx906 FlashAttention Q8_0 extension.

Запускать в контейнере mixa3607/vllm-gfx906 (или аналогичном ROCm + PyTorch).

Использование:
    cd /opt/gfx906_fa/src/extension
    python setup.py build_ext --inplace        # build in-place
    # или:
    pip install -e .                            # install editable

Затем в Python:
    import gfx906_fa
    out = gfx906_fa.forward(q, k_q8, v, scale)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.join(os.path.dirname(HERE), 'kernel')

# gfx906-specific флаги.
# --offload-arch=gfx906 — целевая архитектура MI50 / Vega 20.
# -O3 — оптимизация.
# --save-temps — для отладки (сохраняет .isa/.s во время сборки).
nvcc_flags = [
    '-O3',
    '-std=c++17',
    '--offload-arch=gfx906',
    '-DGGML_USE_HIP',
    '-DGGML_HIP_GFX906',
    '-DFLASH_ATTN_AVAILABLE',
    '-D__HIP_PLATFORM_AMD__',
    # PyTorch cpp_extension принудительно ставит -D__HIP_NO_HALF_OPERATORS__=1
    # (чтобы torch Half не конфликтовал с HIP). Нам нужны native HIP operators
    # для half2 arithmetic в fattn-q8.cuh → undefine их.
    '-U__HIP_NO_HALF_OPERATORS__',
    '-U__HIP_NO_HALF_CONVERSIONS__',
    # Инлайнинг больших __device__ function'ов важен для gfx906:
    '-mllvm', '-amdgpu-function-calls=false',
]

cxx_flags = [
    '-O3',
    '-std=c++17',
    '-DGGML_USE_HIP',
    '-DGGML_HIP_GFX906',
    '-D__HIP_PLATFORM_AMD__',
]

setup(
    name='gfx906_fa',
    version='0.1.0',
    py_modules=['gfx906_fa_backend', 'gfx906_fa_paged'],
    entry_points={
        'vllm.general_plugins': [
            'gfx906_fa = gfx906_fa_backend:register',
        ],
    },
    ext_modules=[
        CUDAExtension(
            name='gfx906_fa',
            sources=[
                os.path.join(HERE, 'gfx906_fa.cpp'),
                os.path.join(HERE, 'gfx906_fa_launcher.cu'),
                os.path.join(HERE, 'gfx906_fa_quant.cu'),
                os.path.join(HERE, 'gfx906_fa_gather.cu'),
            ],
            include_dirs=[
                KERNEL_DIR,
                HERE,
            ],
            extra_compile_args={
                'cxx':  cxx_flags,
                'nvcc': nvcc_flags,  # на ROCm это hipcc
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
