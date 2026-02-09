"""
Lightroom Parameter Mapping Module

This module provides conversion between internal model parameters and Lightroom-style parameters
for better interpretability and compatibility with standard photo editing workflows.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import json


# Lightroom参数范围定义
LIGHTROOM_RANGES = {
    "ExposureFilter": {
        "exposure": (-5.0, 5.0),  # EV stops
        "display_name": "Exposure",
        "unit": "EV"
    },
    "GammaFilter": {
        "gamma": (0.33, 3.0),  # Gamma值
        "display_name": "Gamma",
        "unit": ""
    },
    "ImprovedWhiteBalanceFilter": {
        "wb_r": (0.5, 2.0),  # Red channel multiplier
        "wb_g": (0.5, 2.0),  # Green channel multiplier  
        "wb_b": (0.5, 2.0),  # Blue channel multiplier
        "display_name": "White Balance",
        "unit": ""
    },
    "SaturationFilter": {
        "saturation": (-100, 100),  # Lightroom style -100 to +100
        "display_name": "Saturation",
        "unit": ""
    },
    "ContrastFilter": {
        "contrast": (-100, 100),  # Lightroom style -100 to +100
        "display_name": "Contrast",
        "unit": ""
    },
    "HighlightFilter": {
        "highlight": (-100, 100),  # Lightroom style -100 to +100
        "display_name": "Highlights",
        "unit": ""
    },
    "ShadowFilter": {
        "shadow": (-100, 100),  # Lightroom style -100 to +100
        "display_name": "Shadows",
        "unit": ""
    },
    "SharpenFilter": {
        "sharpen": (0, 150),  # Lightroom style 0 to 150
        "display_name": "Sharpness",
        "unit": ""
    },
    "ColorFilter": {
        # 8 curve steps for 3 channels
        "color_curve": (-10, 10),  # per step adjustment
        "display_name": "Color Curve",
        "unit": ""
    },
    "ToneFilter": {
        # 8 curve steps
        "tone_curve": (-10, 10),  # per step adjustment
        "display_name": "Tone Curve",
        "unit": ""
    }
}


class LightroomParamConverter:
    """转换器：在内部参数和Lightroom参数之间转换"""
    
    def __init__(self, isp_blocks):
        """
        初始化转换器
        
        Args:
            isp_blocks: ISPBlocks实例，包含所有filter的配置
        """
        self.isp_blocks = isp_blocks
        self.filters = isp_blocks.filters if hasattr(isp_blocks, 'filters') else []
        
    def internal_to_lightroom(self, params_list: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        将内部参数转换为Lightroom风格参数
        
        Args:
            params_list: 内部参数列表，每个filter一个tensor
            
        Returns:
            Lightroom风格的参数字典
        """
        lightroom_params = {}
        
        for idx, (filter_obj, param_tensor) in enumerate(zip(self.filters, params_list)):
            filter_name = filter_obj.__class__.__name__
            
            if filter_name not in LIGHTROOM_RANGES:
                # 如果没有定义Lightroom映射，使用内部范围
                lightroom_params[filter_name] = {
                    f"param_{i}": param_tensor[0][i].item() if param_tensor.dim() > 1 else param_tensor[0].item()
                    for i in range(param_tensor.shape[1] if param_tensor.dim() > 1 else 1)
                }
                continue
            
            lr_range_config = LIGHTROOM_RANGES[filter_name]
            filter_params = {}
            
            # 获取内部范围
            internal_min = filter_obj.range_l
            internal_max = filter_obj.range_r
            
            # 转换每个参数
            param_values = param_tensor[0].detach().cpu().numpy() if isinstance(param_tensor, torch.Tensor) else param_tensor[0]
            
            if filter_name == "ExposureFilter":
                # Exposure: 直接使用EV值
                lr_min, lr_max = lr_range_config["exposure"]
                filter_params["exposure"] = self._map_range(
                    param_values[0], internal_min, internal_max, lr_min, lr_max
                )
                
            elif filter_name == "ImprovedWhiteBalanceFilter":
                # White Balance: RGB multipliers
                lr_min, lr_max = lr_range_config["wb_r"]
                filter_params["wb_r"] = param_values[0]  # 已经是multiplier
                filter_params["wb_g"] = param_values[1]
                filter_params["wb_b"] = param_values[2]
                
            elif filter_name in ["SaturationFilter", "ContrastFilter", "HighlightFilter", "ShadowFilter"]:
                # -100 to +100 range
                param_name = list(lr_range_config.keys())[0]
                lr_min, lr_max = lr_range_config[param_name]
                filter_params[param_name] = self._map_range(
                    param_values[0], internal_min, internal_max, lr_min, lr_max
                )
                
            elif filter_name == "SharpenFilter":
                # 0 to 150 range
                lr_min, lr_max = lr_range_config["sharpen"]
                filter_params["sharpen"] = self._map_range(
                    param_values[0], internal_min, internal_max, lr_min, lr_max
                )
                
            elif filter_name in ["ColorFilter", "ToneFilter"]:
                # Curve adjustments
                param_name = list(lr_range_config.keys())[0]
                lr_min, lr_max = lr_range_config[param_name]
                
                if filter_name == "ColorFilter":
                    # 3 channels x 8 steps
                    for ch_idx, ch_name in enumerate(['red', 'green', 'blue']):
                        for step in range(8):
                            idx = ch_idx * 8 + step
                            if idx < len(param_values):
                                filter_params[f"{ch_name}_curve_{step}"] = self._map_range(
                                    param_values[idx], internal_min, internal_max, lr_min, lr_max
                                )
                else:
                    # 8 steps
                    for step in range(min(8, len(param_values))):
                        filter_params[f"tone_curve_{step}"] = self._map_range(
                            param_values[step], internal_min, internal_max, lr_min, lr_max
                        )
            
            lightroom_params[filter_name] = filter_params
            
        return lightroom_params
    
    def lightroom_to_internal(self, lightroom_params: Dict[str, Dict[str, float]]) -> List[torch.Tensor]:
        """
        将Lightroom参数转换回内部参数
        
        Args:
            lightroom_params: Lightroom风格参数字典
            
        Returns:
            内部参数列表
        """
        params_list = []
        
        for filter_obj in self.filters:
            filter_name = filter_obj.__class__.__name__
            
            if filter_name not in lightroom_params:
                # 使用默认值
                params_list.append(torch.zeros(1, filter_obj.num_filter_parameters))
                continue
            
            lr_params = lightroom_params[filter_name]
            internal_min = filter_obj.range_l
            internal_max = filter_obj.range_r
            
            if filter_name not in LIGHTROOM_RANGES:
                # 直接使用提供的值
                param_values = [lr_params.get(f"param_{i}", 0.0) for i in range(filter_obj.num_filter_parameters)]
            else:
                lr_range_config = LIGHTROOM_RANGES[filter_name]
                param_values = []
                
                if filter_name == "ExposureFilter":
                    lr_min, lr_max = lr_range_config["exposure"]
                    param_values = [self._map_range(
                        lr_params.get("exposure", 0.0), lr_min, lr_max, internal_min, internal_max
                    )]
                    
                elif filter_name == "ImprovedWhiteBalanceFilter":
                    param_values = [
                        lr_params.get("wb_r", 1.0),
                        lr_params.get("wb_g", 1.0),
                        lr_params.get("wb_b", 1.0)
                    ]
                    
                elif filter_name in ["SaturationFilter", "ContrastFilter", "HighlightFilter", "ShadowFilter"]:
                    param_name = list(lr_range_config.keys())[0]
                    lr_min, lr_max = lr_range_config[param_name]
                    param_values = [self._map_range(
                        lr_params.get(param_name, 0.0), lr_min, lr_max, internal_min, internal_max
                    )]
                    
                elif filter_name == "SharpenFilter":
                    lr_min, lr_max = lr_range_config["sharpen"]
                    param_values = [self._map_range(
                        lr_params.get("sharpen", 0.0), lr_min, lr_max, internal_min, internal_max
                    )]
                    
                elif filter_name == "ColorFilter":
                    lr_min, lr_max = lr_range_config["color_curve"]
                    for ch_name in ['red', 'green', 'blue']:
                        for step in range(8):
                            val = lr_params.get(f"{ch_name}_curve_{step}", 0.0)
                            param_values.append(self._map_range(val, lr_min, lr_max, internal_min, internal_max))
                            
                elif filter_name == "ToneFilter":
                    lr_min, lr_max = lr_range_config["tone_curve"]
                    for step in range(8):
                        val = lr_params.get(f"tone_curve_{step}", 0.0)
                        param_values.append(self._map_range(val, lr_min, lr_max, internal_min, internal_max))
            
            params_tensor = torch.tensor([param_values], dtype=torch.float32)
            params_list.append(params_tensor)
            
        return params_list
    
    @staticmethod
    def _map_range(value: float, from_min: float, from_max: float, 
                   to_min: float, to_max: float) -> float:
        """线性映射函数"""
        # 归一化到[0, 1]
        normalized = (value - from_min) / (from_max - from_min + 1e-8)
        # 映射到目标范围
        return to_min + normalized * (to_max - to_min)


def save_params_lightroom(params_list, isp_blocks, save_path: str, 
                          include_internal: bool = False):
    """
    以Lightroom风格保存参数
    
    Args:
        params_list: 内部参数列表
        isp_blocks: ISPBlocks实例
        save_path: 保存路径
        include_internal: 是否同时保存内部参数
    """
    converter = LightroomParamConverter(isp_blocks)
    lr_params = converter.internal_to_lightroom(params_list)
    
    # 添加显示名称和单位
    formatted_params = {}
    for filter_name, params in lr_params.items():
        if filter_name in LIGHTROOM_RANGES:
            display_info = {
                "display_name": LIGHTROOM_RANGES[filter_name]["display_name"],
                "unit": LIGHTROOM_RANGES[filter_name].get("unit", ""),
                "parameters": params
            }
        else:
            display_info = {"parameters": params}
        formatted_params[filter_name] = display_info
    
    output = {"lightroom_params": formatted_params}
    
    if include_internal:
        internal_params = {}
        for idx, (filter_obj, param) in enumerate(zip(isp_blocks.filters, params_list)):
            filter_name = filter_obj.__class__.__name__
            if isinstance(param, torch.Tensor):
                internal_params[filter_name] = param[0].detach().cpu().tolist()
            else:
                internal_params[filter_name] = param[0] if hasattr(param, '__getitem__') else [param]
        output["internal_params"] = internal_params
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_params_lightroom(load_path: str, isp_blocks) -> List[torch.Tensor]:
    """
    从Lightroom风格的参数文件加载参数
    
    Args:
        load_path: 参数文件路径
        isp_blocks: ISPBlocks实例
        
    Returns:
        内部参数列表
    """
    with open(load_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取Lightroom参数
    if "lightroom_params" in data:
        lr_params = {}
        for filter_name, info in data["lightroom_params"].items():
            lr_params[filter_name] = info.get("parameters", info)
    else:
        lr_params = data
    
    converter = LightroomParamConverter(isp_blocks)
    return converter.lightroom_to_internal(lr_params)


def print_params_comparison(params_list, isp_blocks):
    """
    打印内部参数和Lightroom参数的对比
    
    Args:
        params_list: 内部参数列表
        isp_blocks: ISPBlocks实例
    """
    converter = LightroomParamConverter(isp_blocks)
    lr_params = converter.internal_to_lightroom(params_list)
    
    print("\n" + "="*80)
    print("参数对比：内部参数 vs Lightroom参数")
    print("="*80)
    
    for idx, (filter_obj, param_tensor) in enumerate(zip(isp_blocks.filters, params_list)):
        filter_name = filter_obj.__class__.__name__
        
        print(f"\n【{filter_name}】")
        print(f"  内部范围: [{filter_obj.range_l:.3f}, {filter_obj.range_r:.3f}]")
        
        # 打印内部参数
        internal_values = param_tensor[0].detach().cpu().numpy() if isinstance(param_tensor, torch.Tensor) else param_tensor[0]
        print(f"  内部参数: {internal_values}")
        
        # 打印Lightroom参数
        if filter_name in lr_params:
            print(f"  Lightroom参数:")
            for param_name, param_value in lr_params[filter_name].items():
                if filter_name in LIGHTROOM_RANGES:
                    unit = LIGHTROOM_RANGES[filter_name].get("unit", "")
                    print(f"    {param_name}: {param_value:.3f} {unit}")
                else:
                    print(f"    {param_name}: {param_value:.3f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # 示例用法
    print("Lightroom参数映射模块")
    print("\n支持的Filter和对应的Lightroom参数范围：")
    for filter_name, config in LIGHTROOM_RANGES.items():
        print(f"\n{filter_name}:")
        print(f"  显示名称: {config['display_name']}")
        for param_name, param_range in config.items():
            if param_name not in ['display_name', 'unit']:
                print(f"  {param_name}: {param_range}")
