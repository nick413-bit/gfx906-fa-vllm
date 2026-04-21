# SPDX-License-Identifier: Apache-2.0
"""
vLLM v1 attention backend для gfx906 (MI50) на базе портированного
Q8 FlashAttention kernel из llama.cpp-gfx906.

Регистрируется как AttentionBackendEnum.CUSTOM:

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum, register_backend,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "gfx906_fa_backend.Gfx906FABackend",
    )

Layout KV-cache совпадает с TritonAttentionBackend:
    (num_blocks, 2, block_size, num_kv_heads, head_size)
так что его можно свободно переключать через флаг
`VLLM_ATTENTION_BACKEND=CUSTOM` без изменения аллокатора.

MVP: gather paged → contiguous на Python-стороне. Оптимизация
(fused gather-kernel на HIP) — следующий шаг после A/B-замеров.
"""

import os as _os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# Наши модули (должны быть в PYTHONPATH)
from gfx906_fa_paged import forward_paged  # noqa: E402

logger = init_logger(__name__)


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
@dataclass
class Gfx906FAMetadata:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor      # [B+1] int32
    seq_lens: torch.Tensor             # [B]   int32
    block_table: torch.Tensor          # [B, max_num_blocks] int32
    slot_mapping: torch.Tensor         # [num_tokens] int64
    use_cascade: bool = False
    common_prefix_len: int = 0


class Gfx906FAMetadataBuilder(
    AttentionMetadataBuilder[Gfx906FAMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> Gfx906FAMetadata:
        # CUDA Graph capture пока не поддерживаем (MVP).
        return self.build(0, common_attn_metadata)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Gfx906FAMetadata:
        return Gfx906FAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            use_cascade=(common_prefix_len > 0),
            common_prefix_len=common_prefix_len,
        )


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------
class Gfx906FABackend(AttentionBackend):
    accept_output_buffer: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
    ]
    # MVP: поддерживаем только FP16 KV (Q8 квантование делается
    # kernel'ом прямо на лету, снаружи — FP16).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "half",
    ]

    # KV write делаем отдельным вызовом (triton_reshape_and_cache_flash)
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        # ВАЖНО: должно совпадать с именем в AttentionBackendEnum, чтобы
        # vllm смог резолвить его через AttentionBackendEnum[name].
        # Регистрируемся как CUSTOM, так что и имя — CUSTOM.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["Gfx906FAImpl"]:
        return Gfx906FAImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # Identical to TritonAttentionBackend → любой бэкенд
        # можно переключать без реаллокации KV.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["Gfx906FAMetadataBuilder"]:
        return Gfx906FAMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Kernel протестирован для 64/128. MiniMax использует 128.
        return head_size in (64, 128)

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # gfx906: major=9, minor=0, patch=6 → capability.to_int() → 906
        # Но в ROCm DeviceCapability не всегда корректно заполняется для gfx906,
        # поэтому пропускаем всё и полагаемся на ручной select.
        return True


# -----------------------------------------------------------------------------
# Impl
# -----------------------------------------------------------------------------
class Gfx906FAImpl(AttentionImpl):

    # ------------------------------------------------------------------
    # CLASS-LEVEL shared gather buffers (K_q8, V_fp16).
    #
    # Внутри одного воркера все attention layers делят эти буфера — kernel
    # читает cache один раз per (seq, head, tok) и пишет в contiguous out
    # per forward вызов. Между forward'ами содержимое не сохраняется.
    #
    # Shared→ значит **одна пара буферов per worker**, а не per layer.
    # Это экономит N_layers × (K_buf + V_buf) VRAM (для MiniMax 60 layers
    # × ~24 MiB = 1.4 GB на одной seq). Grow-логика такая же как у
    # q_pad_buf — при необходимости освобождаем старые перед alloc новых.
    # ------------------------------------------------------------------
    _k_gather_buf: ClassVar[torch.Tensor | None] = None
    _v_gather_buf: ClassVar[torch.Tensor | None] = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("GFX906_FA: alibi_slopes не поддерживается")
        if sliding_window is not None:
            raise NotImplementedError("GFX906_FA: sliding_window не поддерживается")
        if logits_soft_cap not in (None, 0, 0.0):
            raise NotImplementedError("GFX906_FA: logits_soft_cap не поддерживается")
        if sinks is not None:
            raise NotImplementedError("GFX906_FA: sinks не поддерживается")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"GFX906_FA: attn_type={attn_type} не поддерживается"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = num_heads // num_kv_heads

        # ------------------------------------------------------------------
        # Q8_0 side-buffer для K.
        #
        # vLLM аллоцирует основной K/V cache в FP16 (shape получает из
        # get_kv_cache_shape). Параллельно держим side-buffer в block_q8_0
        # формате, чтобы в forward не пересчитывать квантование каждый шаг.
        #
        # Аллокация — lazy в first do_kv_cache_update (тогда уже знаем
        # kv_cache.shape). Live-layout:
        #   [num_blocks, block_size, Hkv, (D/32)*34]  uint8
        # Размер: num_blocks * block_size * Hkv * D * 34/32 байт
        #       = K_fp16_bytes * (34/32) / 2 ≈ 0.53 × K_fp16 bytes.
        # ------------------------------------------------------------------
        self._k_cache_q8: torch.Tensor | None = None
        self._legacy = _os.environ.get("GFX906_FA_LEGACY", "0") == "1"

        # ------------------------------------------------------------------
        # Предаллоцированные буферы для forward_paged.
        # Размеры берутся из vLLM max_num_seqs × max_model_len, но так как
        # в этом контексте их узнать нельзя, используем lazy grow.
        # ------------------------------------------------------------------
        self._q_pad_buf: torch.Tensor | None = None
        # Level 3a: mask_buf убран — inline causal в kernel.

    def fused_output_quant_supported(self, quant_key):
        return False

    # ------------------------------------------------------------------
    # KV cache write (отдельный шаг, как у Triton backend с
    # forward_includes_kv_cache_update=False)
    # ------------------------------------------------------------------
    def _ensure_q8_sidebuffer(self, key_cache: torch.Tensor) -> None:
        """Lazy-аллокация Q8 side-buffer размером совпадающим с K-cache."""
        if self._k_cache_q8 is not None:
            return
        # key_cache shape: [num_blocks, block_size, Hkv, D]  fp16
        num_blocks, block_size, Hkv, D = key_cache.shape
        assert D % 32 == 0, f"D={D} must be multiple of 32"
        bytes_per_row = (D // 32) * 34
        self._k_cache_q8 = torch.empty(
            (num_blocks, block_size, Hkv, bytes_per_row),
            dtype=torch.uint8,
            device=key_cache.device,
        )
        # Обнулим явно — в Q8_0 нули будут декодироваться как 0.
        self._k_cache_q8.zero_()

    def _ensure_forward_buffers(
        self,
        num_seqs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Lazy/grow аллокация q_pad буфера.

        Level 3a: mask_buf удалён — причинность inline в kernel.
        Для prefill 60K это экономит ~480 MB fp16 маски.
        """
        if   max_seqlen_q >  32: ncols1 = 64
        elif max_seqlen_q >  16: ncols1 = 32
        elif max_seqlen_q >   8: ncols1 = 16
        elif max_seqlen_q >   4: ncols1 = 8
        elif max_seqlen_q >   2: ncols1 = 4
        else:                    ncols1 = 2
        Sq_pad = ((max_seqlen_q + ncols1 - 1) // ncols1) * ncols1

        # Q buffer: [B, Hq, Sq_pad, D] query-dtype.
        # ВАЖНО: при grow-реаллокации СНАЧАЛА освобождаем старый tensor
        # (иначе пик VRAM = 2×buf и на prefill 4k+ словим OOM). torch GC
        # рекламирует free сразу после выхода reference count → 0.
        if (self._q_pad_buf is None
                or self._q_pad_buf.shape[0] < num_seqs
                or self._q_pad_buf.shape[2] < Sq_pad
                or self._q_pad_buf.dtype != dtype):
            cur = self._q_pad_buf
            new_shape = (
                max(num_seqs, cur.shape[0] if cur is not None else 0),
                self.num_heads,
                max(Sq_pad, cur.shape[2] if cur is not None else 0),
                self.head_size,
            )
            self._q_pad_buf = None
            del cur
            torch.cuda.empty_cache()
            self._q_pad_buf = torch.empty(new_shape, dtype=dtype, device=device)

    @classmethod
    def _ensure_gather_buffers(
        cls,
        num_seqs: int,
        num_kv_heads: int,
        max_seqlen_k: int,
        head_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared между всеми layers одного воркера: K_q8 + V_fp16 gather buffers.

        shape:
            K: [B, Hkv, Sk_pad, (D/32)*34]  uint8
            V: [B, Hkv, Sk_pad, D]          fp16

        Grow-strategy: free → alloc, чтобы peak VRAM = 1× (не 2×).
        Возвращает буфера ТОЧНОГО запрошенного shape (может быть меньше
        pre-alloc'd если batch сократился — в этом случае slice+.contiguous()
        сделал бы копию, поэтому просто шринкаем буфер до точного размера.
        На практике num_seqs стабилен после warm-up).
        """
        Sk_pad = ((max_seqlen_k + 31) // 32) * 32
        bytes_per_row = (head_size // 32) * 34

        need_realloc = False
        if cls._k_gather_buf is None:
            need_realloc = True
        else:
            b = cls._k_gather_buf
            if (b.shape[0] != num_seqs or b.shape[1] != num_kv_heads
                or b.shape[2] != Sk_pad or b.shape[3] != bytes_per_row
                or b.device != device):
                need_realloc = True

        if need_realloc:
            cls._k_gather_buf = None
            cls._v_gather_buf = None
            torch.cuda.empty_cache()
            cls._k_gather_buf = torch.empty(
                (num_seqs, num_kv_heads, Sk_pad, bytes_per_row),
                dtype=torch.uint8, device=device,
            )
            cls._v_gather_buf = torch.empty(
                (num_seqs, num_kv_heads, Sk_pad, head_size),
                dtype=torch.float16, device=device,
            )
        return cls._k_gather_buf, cls._v_gather_buf

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = kv_cache.unbind(1)

        # 1) Основной FP16 write — vLLM-стандартный путь для V (и legacy K).
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        # 2) Параллельно в Q8 side-buffer для K (fast-path forward).
        #    В legacy-режиме пропускаем — forward будет квантовать на лету.
        # ВАЖНО: Q8 side-buffer путь (LEGACY=0) неконсистентен, когда fp16 kv_cache
        # содержит данные, записанные МИМО нашего do_kv_cache_update (warmup/
        # profile_run/dummy forward'ы vLLM, torch.compile captures и т.п.).
        # В этом случае _k_cache_q8 отстаёт от fp16 cache → forward читает Q8=0
        # и производит garbage output. Reproducer: test_backend_vs_legacy.py.
        # До фикса держим LEGACY=1 (inline-quantize) как дефолт.
        if not self._legacy:
            self._ensure_q8_sidebuffer(key_cache)
            import gfx906_fa
            gfx906_fa.reshape_and_cache_q8(
                key.contiguous() if not key.is_contiguous() else key,
                slot_mapping.to(torch.int64) if slot_mapping.dtype != torch.int64 else slot_mapping,
                self._k_cache_q8,
            )

    def fused_rope_kvcache_supported(self):
        return False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,      # [num_tokens, num_heads, head_size]
        key: torch.Tensor,        # [num_tokens, num_kv_heads, head_size]  (уже записаны в kv_cache через do_kv_cache_update)
        value: torch.Tensor,
        kv_cache: torch.Tensor,   # [num_blocks, 2, block_size, num_kv_heads, head_size]
        attn_metadata: Gfx906FAMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("GFX906_FA: output quantization не поддерживается")

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert not attn_metadata.use_cascade, "GFX906_FA: cascade не поддерживается"

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Unbind KV cache: (..., 2, ...) → (K, V) each [num_blocks, block_size, Hkv, D]
        key_cache, value_cache = kv_cache.unbind(1)

        # Наш forward_paged ждёт: query [num_tokens, Hq, D] fp32
        q_actual = query[:num_actual_tokens]
        if q_actual.dtype != torch.float32:
            q_actual = q_actual.float()
        out_actual = output[:num_actual_tokens]

        # Lazy-grow forward-буферов для q_pad / mask.
        num_seqs = attn_metadata.seq_lens.shape[0]
        self._ensure_forward_buffers(
            num_seqs=num_seqs,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            device=query.device,
            dtype=q_actual.dtype,
        )

        # Level 1 fused-gather буфера (class-level shared между layers).
        # Только когда идём по fused-пути (key_cache_q8 присутствует и не legacy).
        if not self._legacy and self._k_cache_q8 is not None:
            k_gather_buf, v_gather_buf = self._ensure_gather_buffers(
                num_seqs=num_seqs,
                num_kv_heads=self.num_kv_heads,
                max_seqlen_k=attn_metadata.max_seq_len,
                head_size=self.head_size,
                device=query.device,
            )
        else:
            k_gather_buf = v_gather_buf = None

        # forward_paged возвращает [num_tokens, Hq*D] float32.
        # Fast-path: передаём Q8 side-buffer если он есть.
        out_flat = forward_paged(
            query=q_actual,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            scale=self.scale,
            key_cache_q8=self._k_cache_q8 if not self._legacy else None,
            q_pad_buf=self._q_pad_buf,
            mask_buf=None,  # Level 3a: inline causal — mask_buf больше не нужен
            k_gather_buf=k_gather_buf,
            v_gather_buf=v_gather_buf,
        )  # [num_tokens, Hq*D] fp32

        # Результат пишем в output in-place (он [num_tokens, Hq, D] или
        # [num_tokens, Hq*D] — зависит от вызывающего кода).
        out_view = out_actual.view(num_actual_tokens, -1)
        out_view.copy_(out_flat.to(out_view.dtype))

        return output


# -----------------------------------------------------------------------------
# Auto-register как CUSTOM при импорте
# -----------------------------------------------------------------------------
def register() -> None:
    """Регистрирует Gfx906FABackend как AttentionBackendEnum.CUSTOM.

    Вызвать либо вручную в user-коде, либо автоматически через entry point.
    """
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        f"{__name__}.Gfx906FABackend",
    )
    logger.info("GFX906_FA backend registered as AttentionBackendEnum.CUSTOM")


# Авто-регистрация при импорте модуля
register()
