# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Tuple
from types import SimpleNamespace

try:
    import ray
    _ray_available = True
except ImportError:
    ray = None
    _ray_available = False
import torch
import triton
from common_utils import (
    BenchmarkConfig,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    save_configs,
    sort_config,
)
try:
    from ray.experimental.tqdm_ray import tqdm
except Exception:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        def tqdm(x):  # type: ignore
            return x

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.utils import is_hip

_is_hip = is_hip()
set_global_server_args_for_scheduler(SimpleNamespace(enable_deterministic_inference=False))


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: List[int] = None,
    num_iters: int = 100,
) -> float:
    device = torch.device("cuda")
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
            device=device,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
            device=device,
        )
    elif use_int4_w4a16:
        # int4 weights are packed into uint8 with K dimension halved.
        w1 = torch.randint(
            0,
            256,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size // 2,
            ),
            dtype=torch.uint8,
            device=device,
        )
        w2 = torch.randint(
            0,
            256,
            (
                num_experts,
                hidden_size,
                (shard_intermediate_size // 2) // 2,
            ),
            dtype=torch.uint8,
            device=device,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype, device=device
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype, device=device
        )
    gating_output = torch.randn(
        num_iters, num_tokens, num_experts, dtype=torch.float32, device=device
    )

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32, device=device
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32, device=device)
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32, device=device
            )
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            w2_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            a1_scale = torch.randn(1, dtype=torch.float32, device=device)
            a2_scale = torch.randn(1, dtype=torch.float32, device=device)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32, device=device
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32, device=device
            )
    if use_int4_w4a16:
        assert block_shape is not None and block_shape[1] > 0
        group_size = block_shape[1]
        w1_scale = torch.rand(
            (num_experts, shard_intermediate_size, hidden_size // group_size),
            dtype=torch.float32,
            device=device,
        )
        w2_scale = torch.rand(
            (
                num_experts,
                hidden_size,
                (shard_intermediate_size // 2) // group_size,
            ),
            dtype=torch.float32,
            device=device,
        )

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
    )
    topk_output = select_experts(x, input_gating, topk_config)

    def prepare(i: int):
        input_gating = gating_output[i]
        new_topk_output = select_experts(x, input_gating, topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        moe_runner_config = MoeRunnerConfig(
            inplace=True,
        )

        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                topk_output,
                moe_runner_config=moe_runner_config,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    if _is_hip:
        # CUDAGraph on ROCm can be unstable for this workload; use direct timing.
        for _ in range(5):
            run()
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        for i in range(num_iters):
            prepare(i)
            start_events[i].record()
            run()
            end_events[i].record()
        torch.cuda.synchronize()

        latencies: List[float] = []
        for i in range(num_iters):
            latencies.append(start_events[i].elapsed_time(end_events[i]))
        avg = sum(latencies) / num_iters * 1000  # us
        return avg

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    # Flush L2 cache with 256 MB data
    cache_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache_flush.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        prepare(i)
        start_events[i].record()
        graph.replay()
        end_events[i].record()
    torch.cuda.synchronize()

    latencies: List[float] = []
    for i in range(num_iters):
        latencies.append(start_events[i].elapsed_time(end_events[i]))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU.
        if ray is not None:
            self.device_id = int(ray.get_gpu_ids()[0])
        else:
            self.device_id = torch.cuda.current_device()

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
    ) -> Tuple[Dict[str, int], float]:
        torch.cuda.manual_seed_all(0)
        dtype_str = get_config_dtype_str(
            dtype,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
        )
        config_n = shard_intermediate_size // 4 if use_int4_w4a16 else shard_intermediate_size // 2
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        op_config = get_moe_configs(
            num_experts,
            config_n,
            dtype_str,
            block_n,
            block_k,
            per_channel_quant,
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                config_n,
                hidden_size,
                topk,
                dtype_str,
                False,
                block_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            kernel_time = benchmark_config(
                config,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                per_channel_quant,
                block_shape,
            )
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
        search_space: List[Dict[str, int]],
    ) -> Dict[str, int]:
        best_config = None
        best_time = float("inf")
        first_err: Exception | None = None
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            for config in tqdm(search_space):
                try:
                    kernel_time = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        use_int4_w4a16,
                        per_channel_quant,
                        block_shape,
                        num_iters=10,
                    )
                except (triton.runtime.autotuner.OutOfResources, RuntimeError, Exception) as e:
                    # Some configurations may be invalid and fail to compile.
                    if first_err is None:
                        first_err = e
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        if best_config is None:
            raise RuntimeError(f"All configs failed; first error: {first_err}")
        return best_config


if _ray_available:
    BenchmarkWorker = ray.remote(num_gpus=1)(BenchmarkWorker)


def main(args: argparse.Namespace):
    print(args)

    model_config = get_model_config(
        args.model, args.tp_size, args.ep_size, args.disable_shared_experts_fusion
    )

    E = model_config["num_experts"]
    topk = model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]
    if dtype == torch.bfloat16:
        dtype = torch.float16

    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int4_w4a16 = args.dtype == "int4_w4a16"
    per_channel_quant = args.per_channel_quant

    if args.batch_size is None:
        batch_sizes = get_default_batch_sizes()
    else:
        batch_sizes = [args.batch_size]

    if _ray_available:
        ray.init()
        num_gpus = int(ray.available_resources()["GPU"])
        workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]
    else:
        num_gpus = 1
        workers = [BenchmarkWorker(args.seed)]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            if _ray_available:
                output = worker_method.remote(*input_args)
                outputs.append(output)
            else:
                outputs.append(worker_method(*input_args))
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs) if _ray_available else outputs

    if args.tune:
        search_space = get_configs_compute_bound()
        if block_shape is not None:
            block_n, block_k = block_shape[0], block_shape[1]
            search_space = [
                config
                for config in search_space
                if block_k % config["BLOCK_SIZE_K"] == 0
            ]
        if args.max_configs is not None:
            search_space = search_space[: args.max_configs]

        filename = get_config_filename(
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
        )
        print(
            f"Start tuning over {len(search_space)} configurations to create {filename}..."
        )

        start = time.perf_counter()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                    search_space,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            filename,
        )
        end = time.perf_counter()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int4_w4a16"],
        default="auto",
    )
    parser.add_argument(
        "--per-channel-quant",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--max-configs", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    args = parser.parse_args()

    main(args)
