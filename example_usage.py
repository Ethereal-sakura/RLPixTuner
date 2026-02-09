#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLPixTuner 使用示例
演示如何使用 API 进行图像推理
"""

from rlpixtuner_api import RLPixTunerInference, quick_inference


def example_1_basic_usage():
    """示例1: 基本用法 - 单张图片推理"""
    print("\n" + "="*60)
    print("示例1: 单张图片推理")
    print("="*60)
    
    # 创建推理器
    inferencer = RLPixTunerInference(
        model_path='envs/checkpoints/best_model.zip',
        isp_type='full',  # 使用完整的7滤镜流程
        max_step=10,
        verbose=True
    )
    
    # 推理
    output_image, metrics = inferencer.predict(
        input_image_path='examples/input.jpg',
        target_image_path='examples/target.jpg',
        output_path='results/output.png'
    )
    
    # 打印结果
    print(f"\n结果:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  优化步数: {metrics['steps']}")
    
    # 获取 ISP 参数
    isp_params = inferencer.get_isp_parameters()
    print(f"\nISP 参数:")
    for filter_name, param_value in isp_params.items():
        print(f"  {filter_name}: {param_value.flatten()}")


def example_2_batch_inference():
    """示例2: 批量推理"""
    print("\n" + "="*60)
    print("示例2: 批量推理")
    print("="*60)
    
    # 准备图像对列表
    image_pairs = [
        ('examples/img1_input.jpg', 'examples/img1_target.jpg'),
        ('examples/img2_input.jpg', 'examples/img2_target.jpg'),
        ('examples/img3_input.jpg', 'examples/img3_target.jpg'),
    ]
    
    # 创建推理器
    inferencer = RLPixTunerInference(
        model_path='envs/checkpoints/best_model.zip',
        isp_type='full',
        max_step=10,
        verbose=True
    )
    
    # 批量推理
    results = inferencer.predict_batch(
        image_pairs=image_pairs,
        output_dir='results/batch',
        save_comparison=True  # 保存对比图
    )
    
    # 分析结果
    print("\n批量推理结果:")
    for i, result in enumerate(results):
        if result['success']:
            print(f"  [{i+1}] {result['input_path']}: "
                  f"PSNR={result['metrics']['psnr']:.2f} dB, "
                  f"SSIM={result['metrics']['ssim']:.4f}")
        else:
            print(f"  [{i+1}] {result['input_path']}: 失败 - {result['error']}")


def example_3_quick_inference():
    """示例3: 快速推理（简化接口）"""
    print("\n" + "="*60)
    print("示例3: 快速推理")
    print("="*60)
    
    # 一行代码完成推理
    metrics = quick_inference(
        input_path='examples/input.jpg',
        target_path='examples/target.jpg',
        output_path='results/quick_output.png',
        max_step=10
    )
    
    print(f"\n结果: PSNR={metrics['psnr']:.2f} dB")


def example_4_different_isp():
    """示例4: 使用不同的 ISP 流程"""
    print("\n" + "="*60)
    print("示例4: 不同的 ISP 流程对比")
    print("="*60)
    
    isp_types = ['wb', 'exp-wb-cont', 'full']
    
    for isp_type in isp_types:
        print(f"\n使用 ISP 流程: {isp_type}")
        
        inferencer = RLPixTunerInference(
            model_path='envs/checkpoints/best_model.zip',
            isp_type=isp_type,
            max_step=10,
            verbose=False
        )
        
        output_image, metrics = inferencer.predict(
            input_image_path='examples/input.jpg',
            target_image_path='examples/target.jpg',
            output_path=f'results/output_{isp_type}.png'
        )
        
        print(f"  PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")


def example_5_return_pil_image():
    """示例5: 返回 PIL Image 对象"""
    print("\n" + "="*60)
    print("示例5: 返回 PIL Image")
    print("="*60)
    
    inferencer = RLPixTunerInference(
        model_path='envs/checkpoints/best_model.zip',
        isp_type='full',
        max_step=10,
        verbose=True
    )
    
    # 返回 PIL Image 而不是 numpy array
    output_image, metrics = inferencer.predict(
        input_image_path='examples/input.jpg',
        target_image_path='examples/target.jpg',
        return_numpy=False  # 返回 PIL.Image
    )
    
    # 可以直接使用 PIL Image 的方法
    print(f"\n输出图像类型: {type(output_image)}")
    print(f"图像尺寸: {output_image.size}")
    
    # 保存或进一步处理
    output_image.save('results/output_pil.png')
    
    # 或者进行其他 PIL 操作
    # output_image.show()  # 显示图像
    # output_image.resize((512, 512))  # 调整大小
    # output_image.convert('L')  # 转换为灰度


def example_6_custom_settings():
    """示例6: 自定义设置"""
    print("\n" + "="*60)
    print("示例6: 自定义优化步数和设备")
    print("="*60)
    
    # 使用 CPU（如果没有 GPU）
    inferencer = RLPixTunerInference(
        model_path='envs/checkpoints/best_model.zip',
        isp_type='full',
        max_step=5,  # 使用更少的步数（更快但效果可能略差）
        device='cuda',  # 改为 'cpu' 如果没有 GPU
        verbose=True
    )
    
    output_image, metrics = inferencer.predict(
        input_image_path='examples/input.jpg',
        target_image_path='examples/target.jpg',
        output_path='results/output_custom.png'
    )
    
    print(f"\n结果: PSNR={metrics['psnr']:.2f} dB (使用 {metrics['steps']} 步)")


def example_7_error_handling():
    """示例7: 错误处理"""
    print("\n" + "="*60)
    print("示例7: 错误处理")
    print("="*60)
    
    try:
        # 尝试加载不存在的模型
        inferencer = RLPixTunerInference(
            model_path='non_existent_model.zip',
            isp_type='full'
        )
    except FileNotFoundError as e:
        print(f"捕获到错误: {e}")
    
    try:
        # 尝试使用不支持的 ISP 类型
        inferencer = RLPixTunerInference(
            model_path='envs/checkpoints/best_model.zip',
            isp_type='invalid_type'
        )
    except ValueError as e:
        print(f"捕获到错误: {e}")
    
    try:
        # 尝试推理不存在的图像
        inferencer = RLPixTunerInference(
            model_path='envs/checkpoints/best_model.zip',
            isp_type='full'
        )
        output, metrics = inferencer.predict(
            'non_existent_input.jpg',
            'non_existent_target.jpg'
        )
    except FileNotFoundError as e:
        print(f"捕获到错误: {e}")


def main():
    """运行所有示例"""
    print("\n" + "#"*60)
    print("# RLPixTuner API 使用示例")
    print("#"*60)
    
    import os
    
    # 检查模型是否存在
    if not os.path.exists('envs/checkpoints/best_model.zip'):
        print("\n[警告] 预训练模型不存在: envs/checkpoints/best_model.zip")
        print("请先下载或训练模型")
        return
    
    # 创建示例输出目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('examples', exist_ok=True)
    
    # 运行示例（根据需要注释掉不需要的示例）
    try:
        # example_1_basic_usage()
        # example_2_batch_inference()
        # example_3_quick_inference()
        # example_4_different_isp()
        # example_5_return_pil_image()
        # example_6_custom_settings()
        example_7_error_handling()
        
    except Exception as e:
        print(f"\n[错误] 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "#"*60)
    print("# 示例运行完成")
    print("#"*60)
    print("\n提示: 取消注释 main() 中的示例函数来运行相应的示例")


if __name__ == '__main__':
    main()
