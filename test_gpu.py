#!/usr/bin/env python3
"""
Script de prueba para verificar que PyTorch puede usar la GPU correctamente
"""

import torch
import torch.nn as nn
import time

def test_gpu_setup():
    print("PRUEBA DE CONFIGURACIÓN GPU")
    
    # Verificar CUDA
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Memoria libre: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo seleccionado: {device}")
    
    # Crear tensor de prueba
    print("\n" + "-"*40)
    print("PRUEBA DE OPERACIONES GPU")
    print("-"*40)
    
    # Tensor grande para probar GPU
    size = 2000
    print(f"Creando tensores de {size}x{size}...")
    
    # CPU
    start_time = time.time()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"Tiempo CPU: {cpu_time:.3f}s")
    
    if torch.cuda.is_available():
        # GPU
        start_time = time.time()
        a_gpu = torch.randn(size, size, device=device)
        b_gpu = torch.randn(size, size, device=device)
        torch.cuda.synchronize()  # Asegurar que la GPU termine
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"Tiempo GPU: {gpu_time:.3f}s")
        print(f"Aceleración: {cpu_time/gpu_time:.1f}x más rápido")
        
        # Verificar memoria GPU
        print(f"Memoria GPU usada: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
    # Probar red neuronal simple
    print("\n" + "-"*40)
    print("PRUEBA DE RED NEURONAL")
    print("-"*40)
    
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).to(device)
    
    print(f"Modelo en dispositivo: {next(model.parameters()).device}")
    
    # Datos de prueba
    x = torch.randn(32, 1000, device=device)
    y = model(x)
    print(f"Forward pass exitoso! Output shape: {y.shape}")
    print(f"Output en dispositivo: {y.device}")
    
# if __name__ == "__main__":
#     test_gpu_setup()