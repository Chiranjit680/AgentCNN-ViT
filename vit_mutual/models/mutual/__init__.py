import logging
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Dict, Union

import torch
import torch.nn as nn

from cv_lib.config_parsing import get_cfg
from cv_lib.utils import MidExtractor

from vit_mutual.models import get_model
from .base_mutual_model import MutualModel
from .joint_model import JointModel


def get_base_mutual_model(model_cfg: Dict[str, Any], num_classes: int, 
                         task_type: str = "segmentation", 
                         output_stride: int = 8,
                         input_size: tuple = (512, 512)) -> MutualModel:
    """
    Build base mutual model for segmentation task.
    
    Args:
        model_cfg: Model configuration dictionary
        num_classes: Number of segmentation classes
        task_type: Task type ("segmentation" or "classification")
        output_stride: Output stride for segmentation (8, 16, or 32)
        input_size: Input image size (height, width)
    """
    logger = logging.getLogger("get_base_mutual_model")
    names = sorted(model_cfg.keys())
    models = nn.ModuleDict()
    extractors = OrderedDict()
    
    for name in names:
        cfg = model_cfg[name]
        m_cfg = get_cfg(cfg["cfg_path"])["model"]
        
        # Modify model config for segmentation
        if task_type == "segmentation":
            m_cfg = _adapt_config_for_segmentation(m_cfg, num_classes, output_stride, input_size)
        
        model = get_model(m_cfg, num_classes, False, task_type=task_type)
        logger.info("Built submodel: %s for %s", name, task_type)
        
        # Load from checkpoint
        ckpt_path = cfg.get("ckpt", None)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # Handle potential size mismatch for segmentation head
            if task_type == "segmentation":
                model_state = ckpt["model"]
                model_state = _adapt_checkpoint_for_segmentation(model_state, model.state_dict())
                model.load_state_dict(model_state, strict=False)  # Use strict=False for head mismatch
            else:
                model.load_state_dict(ckpt["model"])
            
            logger.info("Loaded ckpt for submodel: %s from dir: %s", name, ckpt_path)
        
        models[name] = model
        
        # Extract features from appropriate layers for segmentation
        extract_layers = cfg.get("extract_layers", cfg.get("extract_layers_segmentation", []))
        extractor = MidExtractor(model, extract_layers)
        extractors[name] = extractor
    
    return MutualModel(models, extractors, task_type=task_type)


def get_joint_model(model_cfg: Dict[str, Any], num_classes: int,
                   task_type: str = "segmentation",
                   output_stride: int = 8,
                   input_size: tuple = (512, 512)) -> JointModel:
    """
    Build joint model for segmentation task.
    
    Args:
        model_cfg: Model configuration dictionary
        num_classes: Number of segmentation classes
        task_type: Task type ("segmentation" or "classification")
        output_stride: Output stride for segmentation
        input_size: Input image size (height, width)
    """
    # Get ViT and CNN configurations
    vit_cfg = model_cfg["vit"]
    cnn_cfg = model_cfg["cnn"]
    
    # Adapt ViT config for segmentation
    if task_type == "segmentation":
        vit_cfg = _adapt_vit_config_for_segmentation(vit_cfg, num_classes, output_stride, input_size)
    
    vit = get_model(
        model_cfg=vit_cfg,
        num_classes=num_classes,
        with_wrapper=False,
        task_type=task_type
    )
    
    joint_model = JointModel(
        vit=vit,
        input_proj_cfg=cnn_cfg["input_proj"],
        norm_cfg=cnn_cfg["norm"],
        bias=True,
        extract_cnn=model_cfg["extract_layers_cnn"],
        extract_vit=model_cfg["extract_layers_vit"],
        down_sample_layers=cnn_cfg.get("down_sample_layers", list()),
        cnn_pre_norm=cnn_cfg["pre_norm"],
        task_type=task_type,
        output_stride=output_stride,
        input_size=input_size,
        **vit_cfg["transformer"],
    )
    return joint_model


def _adapt_config_for_segmentation(config: Dict[str, Any], num_classes: int, 
                                 output_stride: int, input_size: tuple) -> Dict[str, Any]:
    """Adapt model configuration for segmentation task."""
    config = deepcopy(config)
    
    # Add segmentation-specific configurations
    config["task_type"] = "segmentation"
    config["num_classes"] = num_classes
    config["output_stride"] = output_stride
    config["input_size"] = input_size
    
    # Modify head configuration for dense prediction
    if "head" in config:
        config["head"]["type"] = "segmentation_head"
        config["head"]["num_classes"] = num_classes
        config["head"]["output_stride"] = output_stride
    
    # Add decoder configuration if not present
    if "decoder" not in config:
        config["decoder"] = {
            "type": "simple_decoder",
            "in_channels": config.get("embed_dim", 768),
            "num_classes": num_classes,
            "output_stride": output_stride
        }
    
    return config


def _adapt_vit_config_for_segmentation(vit_cfg: Dict[str, Any], num_classes: int,
                                     output_stride: int, input_size: tuple) -> Dict[str, Any]:
    """Adapt ViT configuration for segmentation."""
    vit_cfg = deepcopy(vit_cfg)
    
    # Modify patch size and stride for dense prediction
    if output_stride == 8:
        vit_cfg["patch_size"] = 8
    elif output_stride == 16:
        vit_cfg["patch_size"] = 16
    else:
        vit_cfg["patch_size"] = 32
    
    # Add segmentation head
    vit_cfg["head"] = {
        "type": "segmentation_head",
        "num_classes": num_classes,
        "output_stride": output_stride
    }
    
    return vit_cfg


def _adapt_checkpoint_for_segmentation(checkpoint_state: Dict[str, torch.Tensor],
                                     model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Adapt checkpoint state dict for segmentation by handling classifier head mismatch.
    """
    adapted_state = {}
    
    for key, value in checkpoint_state.items():
        if key in model_state:
            # Check if dimensions match
            if value.shape == model_state[key].shape:
                adapted_state[key] = value
            else:
                # Skip mismatched layers (usually the classification head)
                logging.warning(f"Skipping layer {key} due to shape mismatch: "
                              f"checkpoint {value.shape} vs model {model_state[key].shape}")
        else:
            # Skip keys not in model (e.g., old classification head)
            logging.warning(f"Skipping key {key} not found in model")
    
    return adapted_state


__REGISTERED_MUTUAL_MODEL__ = {
    "base_mutual": get_base_mutual_model,
    "joint_model": get_joint_model
}


def get_mutual_model(mutual_model_cfg: Dict[str, Any], num_classes: int,
                    task_type: str = "segmentation",
                    output_stride: int = 8,
                    input_size: tuple = (512, 512)) -> Union[MutualModel, JointModel]:
    """
    Get mutual model for specified task.
    
    Args:
        mutual_model_cfg: Mutual model configuration
        num_classes: Number of classes (for segmentation: number of semantic classes)
        task_type: Task type ("segmentation" or "classification")
        output_stride: Output stride for segmentation (8, 16, or 32)
        input_size: Input image size (height, width)
    """
    mutual_model_cfg = deepcopy(mutual_model_cfg)
    name = mutual_model_cfg.pop("name")
    
    # Add task-specific parameters
    if task_type == "segmentation":
        model = __REGISTERED_MUTUAL_MODEL__[name](
            mutual_model_cfg, num_classes, task_type, output_stride, input_size
        )
    else:
        model = __REGISTERED_MUTUAL_MODEL__[name](mutual_model_cfg, num_classes)
    
    return model