"""
模型优化和部署
学习目标：掌握深度学习模型的优化、压缩和部署技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

print("=== 模型优化和部署 ===\n")

# 1. 模型优化理论
print("1. 模型优化理论")
print("模型优化目标：")
print("- 减少模型大小：存储和传输效率")
print("- 提高推理速度：实时应用需求")
print("- 降低能耗：移动设备友好")
print("- 保持精度：性能与效率平衡")

print("\n优化技术分类：")
print("1. 模型压缩：剪枝、量化、知识蒸馏")
print("2. 架构优化：MobileNet、EfficientNet")
print("3. 推理优化：算子融合、内存优化")
print("4. 硬件加速：GPU、TPU、专用芯片")

# 2. 基础模型定义
print("\n2. 基础模型定义")

class BaselineCNN(nn.Module):
    """基础CNN模型"""
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EfficientCNN(nn.Module):
    """优化后的CNN模型"""
    def __init__(self, num_classes=10, width_multiplier=1.0):
        super(EfficientCNN, self).__init__()
        
        def make_channels(channels):
            return int(channels * width_multiplier)
        
        self.features = nn.Sequential(
            # 深度可分离卷积块
            self._make_depthwise_block(3, make_channels(32)),
            self._make_depthwise_block(make_channels(32), make_channels(64), stride=2),
            self._make_depthwise_block(make_channels(64), make_channels(128), stride=2),
            self._make_depthwise_block(make_channels(128), make_channels(256), stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(make_channels(256), num_classes)
        )
    
    def _make_depthwise_block(self, in_channels, out_channels, stride=1):
        """深度可分离卷积块"""
        return nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # 点卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3. 模型剪枝
print("\n3. 模型剪枝")

class ModelPruner:
    """模型剪枝器"""
    
    def __init__(self, model):
        self.model = model
        self.original_weights = {}
        self.masks = {}
        
    def magnitude_pruning(self, pruning_ratio=0.5):
        """基于权重大小的剪枝"""
        print(f"执行幅度剪枝，剪枝比例: {pruning_ratio}")
        
        # 收集所有权重
        all_weights = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                all_weights.extend(module.weight.data.abs().flatten().tolist())
        
        # 计算阈值
        all_weights.sort()
        threshold_index = int(len(all_weights) * pruning_ratio)
        threshold = all_weights[threshold_index]
        
        # 应用剪枝
        pruned_params = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                mask = module.weight.data.abs() > threshold
                self.masks[name] = mask
                module.weight.data *= mask.float()
                
                pruned_params += (mask == 0).sum().item()
                total_params += mask.numel()
        
        actual_pruning_ratio = pruned_params / total_params
        print(f"实际剪枝比例: {actual_pruning_ratio:.4f}")
        return actual_pruning_ratio
    
    def structured_pruning(self, pruning_ratio=0.3):
        """结构化剪枝（通道剪枝）"""
        print(f"执行结构化剪枝，剪枝比例: {pruning_ratio}")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                # 计算通道重要性（基于L1范数）
                importance = module.weight.data.abs().sum(dim=(1, 2, 3))
                
                # 确定要剪枝的通道数
                num_channels_to_prune = int(module.out_channels * pruning_ratio)
                if num_channels_to_prune > 0:
                    # 找到最不重要的通道
                    _, indices = torch.topk(importance, num_channels_to_prune, largest=False)
                    
                    # 创建掩码
                    mask = torch.ones_like(importance, dtype=torch.bool)
                    mask[indices] = False
                    
                    # 应用掩码
                    module.weight.data[~mask] = 0
                    if module.bias is not None:
                        module.bias.data[~mask] = 0
    
    def gradual_pruning(self, initial_ratio=0.1, final_ratio=0.9, steps=10):
        """逐步剪枝"""
        print(f"执行逐步剪枝，从 {initial_ratio} 到 {final_ratio}，共 {steps} 步")
        
        ratios = np.linspace(initial_ratio, final_ratio, steps)
        
        for i, ratio in enumerate(ratios):
            print(f"步骤 {i+1}: 剪枝比例 {ratio:.3f}")
            self.magnitude_pruning(ratio)
            yield ratio
    
    def get_sparsity(self):
        """计算模型稀疏度"""
        total_params = 0
        zero_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                params = module.weight.data
                total_params += params.numel()
                zero_params += (params == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        return sparsity

# 4. 模型量化
print("\n4. 模型量化")

class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self, model):
        self.model = model
        
    def dynamic_quantization(self):
        """动态量化"""
        print("执行动态量化...")
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        return quantized_model
    
    def static_quantization(self, calibration_loader):
        """静态量化"""
        print("执行静态量化...")
        
        # 设置量化配置
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备模型
        torch.quantization.prepare(self.model, inplace=True)
        
        # 校准
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                if batch_idx >= 100:  # 只使用部分数据校准
                    break
                self.model(data)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        return quantized_model
    
    def fake_quantization(self):
        """伪量化（用于训练）"""
        print("应用伪量化...")
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self.model, inplace=True)
        return self.model

def quantize_weights(weights, bits=8):
    """权重量化函数"""
    # 计算量化范围
    min_val = weights.min()
    max_val = weights.max()
    
    # 量化
    scale = (max_val - min_val) / (2**bits - 1)
    quantized = torch.round((weights - min_val) / scale)
    
    # 反量化
    dequantized = quantized * scale + min_val
    
    return dequantized, scale, min_val

# 5. 知识蒸馏
print("\n5. 知识蒸馏")

class KnowledgeDistillation:
    """知识蒸馏"""
    
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """蒸馏损失函数"""
        # 软目标损失
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # 硬目标损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 组合损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss, hard_loss
    
    def train_student(self, train_loader, epochs=10, lr=0.001):
        """训练学生模型"""
        optimizer = optim.Adam(self.student_model.parameters(), lr=lr)
        self.student_model.train()
        
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_soft_loss = 0
            epoch_hard_loss = 0
            
            for batch_idx, (data, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 学生模型前向传播
                student_logits = self.student_model(data)
                
                # 教师模型前向传播
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)
                
                # 计算损失
                total_loss, soft_loss, hard_loss = self.distillation_loss(
                    student_logits, teacher_logits, labels
                )
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_soft_loss += soft_loss.item()
                epoch_hard_loss += hard_loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, '
                          f'Total Loss: {total_loss.item():.4f}, '
                          f'Soft Loss: {soft_loss.item():.4f}, '
                          f'Hard Loss: {hard_loss.item():.4f}')
            
            avg_loss = epoch_loss / len(train_loader)
            training_losses.append(avg_loss)
            print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
        
        return training_losses

# 6. 神经架构搜索（简化版本）
print("\n6. 神经架构搜索（简化版本）")

class NASSearchSpace:
    """NAS搜索空间"""
    
    def __init__(self):
        self.operations = [
            'conv_3x3',
            'conv_5x5', 
            'depthwise_conv_3x3',
            'depthwise_conv_5x5',
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect'
        ]
    
    def sample_architecture(self, num_layers=6):
        """随机采样一个架构"""
        architecture = []
        for _ in range(num_layers):
            op = np.random.choice(self.operations)
            architecture.append(op)
        return architecture
    
    def build_model_from_arch(self, architecture, input_channels=3, num_classes=10):
        """根据架构描述构建模型"""
        layers = []
        current_channels = input_channels
        
        for i, op in enumerate(architecture):
            if op == 'conv_3x3':
                out_channels = min(64 * (2 ** (i // 2)), 256)
                layers.append(nn.Conv2d(current_channels, out_channels, 3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                current_channels = out_channels
                
            elif op == 'conv_5x5':
                out_channels = min(64 * (2 ** (i // 2)), 256)
                layers.append(nn.Conv2d(current_channels, out_channels, 5, padding=2))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                current_channels = out_channels
                
            elif op == 'depthwise_conv_3x3':
                layers.append(nn.Conv2d(current_channels, current_channels, 3, 
                                      padding=1, groups=current_channels))
                layers.append(nn.BatchNorm2d(current_channels))
                layers.append(nn.ReLU(inplace=True))
                
            elif op == 'max_pool_3x3':
                layers.append(nn.MaxPool2d(3, stride=1, padding=1))
                
            elif op == 'avg_pool_3x3':
                layers.append(nn.AvgPool2d(3, stride=1, padding=1))
                
            elif op == 'skip_connect':
                # 恒等映射
                pass
            
            # 在某些层后添加下采样
            if i in [1, 3]:
                layers.append(nn.MaxPool2d(2, stride=2))
        
        # 添加分类头
        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(current_channels, num_classes)
        ])
        
        return nn.Sequential(*layers)

class EvolutionaryNAS:
    """进化算法NAS"""
    
    def __init__(self, search_space, population_size=20, generations=10):
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.fitness_scores = []
    
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            arch = self.search_space.sample_architecture()
            self.population.append(arch)
    
    def evaluate_fitness(self, architecture, train_loader, test_loader):
        """评估架构适应度"""
        model = self.search_space.build_model_from_arch(architecture)
        
        # 简化训练（只训练少量epoch）
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(2):  # 只训练2个epoch
            for batch_idx, (data, labels) in enumerate(train_loader):
                if batch_idx >= 10:  # 只训练10个batch
                    break
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                if batch_idx >= 5:  # 只评估5个batch
                    break
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy

# 7. 模型部署优化
print("\n7. 模型部署优化")

class ModelOptimizer:
    """模型部署优化器"""
    
    def __init__(self, model):
        self.model = model
    
    def fuse_modules(self):
        """模块融合"""
        print("执行模块融合...")
        
        # 查找可融合的模块
        fused_model = torch.quantization.fuse_modules(
            self.model,
            [['features.0', 'features.1', 'features.2']]  # Conv-BN-ReLU
        )
        return fused_model
    
    def convert_to_torchscript(self, example_input):
        """转换为TorchScript"""
        print("转换为TorchScript...")
        
        self.model.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, example_input)
        
        return traced_model
    
    def convert_to_onnx(self, example_input, output_path):
        """转换为ONNX格式"""
        print(f"转换为ONNX格式，保存到: {output_path}")
        
        self.model.eval()
        torch.onnx.export(
            self.model,
            example_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    def benchmark_model(self, example_input, num_runs=100):
        """性能基准测试"""
        print(f"执行性能基准测试，运行 {num_runs} 次...")
        
        self.model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(example_input)
        
        # 测试
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(example_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        results = {
            'avg_time': avg_time,
            'std_time': std_time,
            'fps': fps,
            'all_times': times
        }
        
        print(f"平均推理时间: {avg_time*1000:.2f} ms")
        print(f"标准差: {std_time*1000:.2f} ms")
        print(f"FPS: {fps:.2f}")
        
        return results

# 8. 模型分析工具
print("\n8. 模型分析工具")

class ModelAnalyzer:
    """模型分析器"""
    
    def __init__(self, model):
        self.model = model
    
    def count_parameters(self):
        """计算参数数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        results = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params
        }
        
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        print(f"非可训练参数数: {total_params - trainable_params:,}")
        
        return results
    
    def calculate_flops(self, input_shape):
        """计算FLOPs（简化版本）"""
        flops = 0
        
        def flop_count_hook(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Conv2d):
                # 卷积层FLOPs
                batch_size = input[0].shape[0]
                output_dims = output.shape[2:]
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                groups = module.groups
                
                filters_per_channel = out_channels // groups
                conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
                
                active_elements_count = batch_size * int(np.prod(output_dims))
                overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
                
                # 加上偏置项
                bias_flops = 0
                if module.bias is not None:
                    bias_flops = out_channels * active_elements_count
                
                overall_flops = overall_conv_flops + bias_flops
                flops += overall_flops
                
            elif isinstance(module, nn.Linear):
                # 全连接层FLOPs
                batch_size = input[0].shape[0]
                flops += batch_size * module.in_features * module.out_features
        
        # 注册钩子
        hooks = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(flop_count_hook))
        
        # 前向传播
        dummy_input = torch.randn(1, *input_shape)
        self.model.eval()
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        print(f"总FLOPs: {flops:,}")
        print(f"总GFLOPs: {flops / 1e9:.2f}")
        
        return flops
    
    def memory_usage(self, input_shape):
        """估算内存使用"""
        # 激活内存
        activation_memory = 0
        
        def memory_hook(module, input, output):
            nonlocal activation_memory
            if isinstance(output, torch.Tensor):
                activation_memory += output.numel() * 4  # 假设float32
        
        # 参数内存
        param_memory = sum(p.numel() * 4 for p in self.model.parameters())
        
        # 注册钩子
        hooks = []
        for module in self.model.modules():
            hooks.append(module.register_forward_hook(memory_hook))
        
        # 前向传播
        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        total_memory = param_memory + activation_memory
        
        print(f"参数内存: {param_memory / 1024**2:.2f} MB")
        print(f"激活内存: {activation_memory / 1024**2:.2f} MB") 
        print(f"总内存: {total_memory / 1024**2:.2f} MB")
        
        return {
            'param_memory': param_memory,
            'activation_memory': activation_memory,
            'total_memory': total_memory
        }

# 9. 实验和比较
print("\n9. 实验和比较")

def compare_optimization_methods():
    """比较不同优化方法"""
    print("比较不同优化方法...")
    
    # 创建示例模型
    original_model = BaselineCNN(num_classes=10)
    efficient_model = EfficientCNN(num_classes=10, width_multiplier=0.5)
    
    # 分析器
    original_analyzer = ModelAnalyzer(original_model)
    efficient_analyzer = ModelAnalyzer(efficient_model)
    
    print("\n=== 原始模型 ===")
    original_params = original_analyzer.count_parameters()
    original_flops = original_analyzer.calculate_flops((3, 32, 32))
    original_memory = original_analyzer.memory_usage((3, 32, 32))
    
    print("\n=== 优化模型 ===")
    efficient_params = efficient_analyzer.count_parameters()
    efficient_flops = efficient_analyzer.calculate_flops((3, 32, 32))
    efficient_memory = efficient_analyzer.memory_usage((3, 32, 32))
    
    # 计算压缩比
    param_compression = original_params['total_params'] / efficient_params['total_params']
    flop_compression = original_flops / efficient_flops
    memory_compression = original_memory['total_memory'] / efficient_memory['total_memory']
    
    print(f"\n=== 压缩效果 ===")
    print(f"参数压缩比: {param_compression:.2f}x")
    print(f"FLOPs压缩比: {flop_compression:.2f}x")
    print(f"内存压缩比: {memory_compression:.2f}x")
    
    return {
        'original': {
            'params': original_params['total_params'],
            'flops': original_flops,
            'memory': original_memory['total_memory']
        },
        'optimized': {
            'params': efficient_params['total_params'],
            'flops': efficient_flops,
            'memory': efficient_memory['total_memory']
        },
        'compression_ratios': {
            'params': param_compression,
            'flops': flop_compression,
            'memory': memory_compression
        }
    }

# 10. 可视化分析
print("\n10. 可视化分析")

def visualize_optimization_results():
    """可视化优化结果"""
    
    # 模拟数据
    methods = ['原始模型', '剪枝', '量化', '蒸馏', '架构优化']
    
    # 模拟指标
    model_sizes = [100, 50, 25, 75, 30]  # MB
    inference_times = [100, 80, 60, 85, 45]  # ms
    accuracies = [92.5, 91.8, 91.2, 92.0, 91.5]  # %
    flops = [1000, 500, 250, 900, 200]  # MFLOPs
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 10.1 模型大小对比
    bars1 = axes[0, 0].bar(methods, model_sizes, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('模型大小对比')
    axes[0, 0].set_ylabel('大小 (MB)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 添加数值标注
    for bar, size in zip(bars1, model_sizes):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{size}', ha='center', va='bottom')
    
    # 10.2 推理时间对比
    bars2 = axes[0, 1].bar(methods, inference_times, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('推理时间对比')
    axes[0, 1].set_ylabel('时间 (ms)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar, time in zip(bars2, inference_times):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{time}', ha='center', va='bottom')
    
    # 10.3 精度对比
    bars3 = axes[0, 2].bar(methods, accuracies, alpha=0.7, color='lightgreen')
    axes[0, 2].set_title('模型精度对比')
    axes[0, 2].set_ylabel('精度 (%)')
    axes[0, 2].set_ylim(90, 93)
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars3, accuracies):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc}', ha='center', va='bottom')
    
    # 10.4 效率-精度权衡
    axes[1, 0].scatter(inference_times, accuracies, s=100, alpha=0.7)
    for i, method in enumerate(methods):
        axes[1, 0].annotate(method, (inference_times[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 0].set_xlabel('推理时间 (ms)')
    axes[1, 0].set_ylabel('精度 (%)')
    axes[1, 0].set_title('效率-精度权衡')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 10.5 压缩比对比
    compression_ratios = [1.0, 2.0, 4.0, 1.33, 3.33]
    bars5 = axes[1, 1].bar(methods, compression_ratios, alpha=0.7, color='gold')
    axes[1, 1].set_title('压缩比对比')
    axes[1, 1].set_ylabel('压缩比 (倍)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for bar, ratio in zip(bars5, compression_ratios):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{ratio:.1f}x', ha='center', va='bottom')
    
    # 10.6 综合性能雷达图
    from math import pi
    
    # 归一化指标
    norm_size = [(100 - size) / 100 for size in model_sizes]  # 越小越好
    norm_time = [(100 - time) / 100 for time in inference_times]  # 越小越好
    norm_acc = [(acc - 90) / 3 for acc in accuracies]  # 越大越好
    
    # 选择一个方法展示
    method_idx = 4  # 架构优化
    values = [norm_size[method_idx], norm_time[method_idx], norm_acc[method_idx]]
    values += values[:1]  # 闭合
    
    angles = [n / 3 * 2 * pi for n in range(3)]
    angles += angles[:1]
    
    axes[1, 2].plot(angles, values, 'o-', linewidth=2, label=methods[method_idx])
    axes[1, 2].fill(angles, values, alpha=0.25)
    axes[1, 2].set_xticks(angles[:-1])
    axes[1, 2].set_xticklabels(['模型大小', '推理速度', '精度'])
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('综合性能雷达图')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    return fig

# 运行实验
comparison_results = compare_optimization_methods()
fig = visualize_optimization_results()
fig.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/model_optimization_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 模型优化和部署总结 ===")
print("✅ 理解模型优化的基本概念")
print("✅ 实现模型剪枝技术")
print("✅ 掌握模型量化方法")
print("✅ 学习知识蒸馏技术")
print("✅ 了解神经架构搜索")
print("✅ 掌握模型部署优化")
print("✅ 分析模型性能指标")

print("\n关键技术:")
print("1. 模型剪枝：结构化、非结构化、逐步剪枝")
print("2. 模型量化：动态量化、静态量化、伪量化")
print("3. 知识蒸馏：软目标、硬目标、温度参数")
print("4. 架构优化：MobileNet、EfficientNet、NAS")
print("5. 推理优化：算子融合、内存优化、并行化")

print("\n优化策略:")
print("1. 压缩优先：剪枝 → 量化 → 蒸馏")
print("2. 精度保持：渐进式优化、微调恢复")
print("3. 硬件适配：针对目标设备优化")
print("4. 端到端：训练、优化、部署一体化")
print("5. 自动化：AutoML、NAS自动搜索")

print("\n部署考虑:")
print("1. 延迟要求：实时性vs批处理")
print("2. 内存限制：移动设备vs服务器")
print("3. 能耗约束：电池续航vs性能")
print("4. 精度需求：可接受的精度损失")
print("5. 更新频率：模型版本管理")

print("\n=== 练习任务 ===")
print("1. 实现更高级的剪枝算法（SNIP、GraSP）")
print("2. 尝试混合精度训练")
print("3. 实现模型并行和流水线并行")
print("4. 构建完整的模型服务系统")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 研究硬件感知的神经架构搜索")
print("2. 实现动态神经网络（Early Exit）")
print("3. 构建联邦学习模型压缩")
print("4. 研究模型量化的理论分析")
print("5. 实现边缘设备的实时推理系统")