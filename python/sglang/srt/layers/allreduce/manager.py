"""
Global manager for adaptive allreduce layer.

This module provides a global singleton to manage the AdaptiveAllReduceLayer instance,
similar to how FlashInferWorkspaceManager works.
"""

import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tp_group
from sglang.srt.layers.allreduce.adaptive_allreduce import AdaptiveAllReduceLayer
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class AdaptiveAllReduceManager:
    """
    Global manager for adaptive allreduce layers.

    This manager creates and maintains AdaptiveAllReduceLayer instances per hidden_size,
    and provides them to model layers during forward pass.
    """

    def __init__(self):
        self.layers = {}  # Dict[hidden_size, AdaptiveAllReduceLayer]
        self.initialized = False
        self.communicators_initialized = False

        # Communicators that will be shared across all layers
        self.torch_symm_mem_communicator = None
        self.custom_allreduce = None

    def initialize_communicators(self):
        """Initialize shared communicators (torch_symm_mem and custom_allreduce)."""
        if self.communicators_initialized:
            return

        server_args = get_global_server_args()
        if not server_args.enable_adaptive_allreduce:
            return

        world_size = get_tensor_model_parallel_world_size()
        if world_size <= 1:
            logger.info(
                "Single GPU, skipping adaptive allreduce communicators initialization"
            )
            return

        device = torch.cuda.current_device()

        from sglang.srt.layers.flashinfer_comm_fusion import (
            ensure_workspace_initialized,
        )

        ensure_workspace_initialized(
            max_token_num=2048,
            hidden_dim=4096,
            use_fp32_lamport=False,
        )
        logger.info("FlashInfer workspace initialized for adaptive allreduce")

        from sglang.srt.distributed.device_communicators.torch_symm_mem import (
            TorchSymmMemCommunicator,
        )

        self.torch_symm_mem_communicator = TorchSymmMemCommunicator(
            group=get_tp_group().device_group,
            device=device,
        )
        if not self.torch_symm_mem_communicator.disabled:
            logger.info(
                "Torch symmetric memory communicator initialized for adaptive allreduce"
            )
        else:
            logger.info("Torch symmetric memory communicator is disabled")
            self.torch_symm_mem_communicator = None

        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.custom_allreduce = CustomAllreduce(
            group=get_tp_group().cpu_group,
            device=device,
        )
        if not self.custom_allreduce.disabled:
            logger.info("Custom allreduce initialized for adaptive allreduce")
        else:
            logger.info("Custom allreduce is disabled")
            self.custom_allreduce = None

        self.communicators_initialized = True

    def get_layer(
        self,
        hidden_size: int,
        flashinfer_workspace_tensor: Optional[torch.Tensor] = None,
    ) -> Optional[AdaptiveAllReduceLayer]:
        """
        Get or create an AdaptiveAllReduceLayer for the given hidden_size.

        Args:
            hidden_size: Hidden dimension
            flashinfer_workspace_tensor: FlashInfer workspace (from FlashInferWorkspaceManager)

        Returns:
            AdaptiveAllReduceLayer instance or None if adaptive allreduce is disabled
        """
        server_args = get_global_server_args()
        if not server_args.enable_adaptive_allreduce:
            return None

        # Initialize communicators if not already done
        if not self.communicators_initialized:
            self.initialize_communicators()

        # Return existing layer if already created
        if hidden_size in self.layers:
            return self.layers[hidden_size]

        # Create new layer
        world_size = get_tensor_model_parallel_world_size()
        if world_size <= 1:
            return None

        layer = AdaptiveAllReduceLayer(
            hidden_size=hidden_size,
            enable_adaptive_allreduce=True,
            flashinfer_workspace_tensor=flashinfer_workspace_tensor,
            torch_symm_mem_communicator=self.torch_symm_mem_communicator,
            custom_allreduce=self.custom_allreduce,
        )

        self.layers[hidden_size] = layer
        logger.info(f"Created AdaptiveAllReduceLayer for hidden_size={hidden_size}")

        return layer

    def cleanup(self):
        """Cleanup all resources."""
        self.layers.clear()

        if self.custom_allreduce is not None:
            try:
                self.custom_allreduce.close()
            except Exception as e:
                logger.warning(f"Failed to cleanup custom allreduce: {e}")

        self.torch_symm_mem_communicator = None
        self.custom_allreduce = None
        self.communicators_initialized = False
        self.initialized = False


# Global singleton instance
_adaptive_allreduce_manager = AdaptiveAllReduceManager()


def get_adaptive_allreduce_layer(
    hidden_size: int,
    flashinfer_workspace_tensor: Optional[torch.Tensor] = None,
) -> Optional[AdaptiveAllReduceLayer]:
    """
    Get adaptive allreduce layer for the given hidden_size.

    This is the main API for models to get the adaptive allreduce layer.

    Args:
        hidden_size: Hidden dimension
        flashinfer_workspace_tensor: FlashInfer workspace (optional)

    Returns:
        AdaptiveAllReduceLayer instance or None if disabled
    """
    return _adaptive_allreduce_manager.get_layer(
        hidden_size=hidden_size,
        flashinfer_workspace_tensor=flashinfer_workspace_tensor,
    )


def cleanup_adaptive_allreduce():
    """Cleanup adaptive allreduce resources."""
    _adaptive_allreduce_manager.cleanup()
