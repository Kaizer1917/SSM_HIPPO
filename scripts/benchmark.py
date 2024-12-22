import argparse
import torch
import onnxruntime
import numpy as np
import time
import logging
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager
import psutil
import GPUtil
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name: str) -> float:
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.debug(f"{name}: {elapsed:.4f} seconds")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark SSM_HIPPO model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model (PyTorch, ONNX, or Quantized)')
    parser.add_argument('--batch_sizes', type=str, default='1,8,16,32',
                       help='Comma-separated list of batch sizes to test')
    parser.add_argument('--sequence_lengths', type=str, default='32,64,128,256',
                       help='Comma-separated list of sequence lengths to test')
    parser.add_argument('--num_features', type=int, default=256,
                       help='Number of input features')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of inference runs for each configuration')
    parser.add_argument('--warmup_runs', type=int, default=10,
                       help='Number of warmup runs before benchmarking')
    parser.add_argument('--output_path', type=str, default='benchmark_results',
                       help='Path to save benchmark results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run benchmarks on')
    return parser.parse_args()

class ModelBenchmarker:
    def __init__(self, model_path: str, device: str):
        self.model_path = Path(model_path)
        self.device = device
        self.model_type = self._determine_model_type()
        self.model = self._load_model()
        
    def _determine_model_type(self) -> str:
        """Determine the type of model based on file extension."""
        suffix = self.model_path.suffix
        if suffix == '.onnx':
            return 'onnx'
        elif suffix == '.pt':
            # Check if model is quantized by loading and checking metadata
            try:
                checkpoint = torch.load(self.model_path)
                return 'quantized' if checkpoint.get('quantized', False) else 'pytorch'
            except:
                return 'pytorch'
        raise ValueError(f"Unsupported model format: {suffix}")

    def _load_model(self) -> Any:
        """Load the model based on its type."""
        try:
            if self.model_type == 'onnx':
                providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                return onnxruntime.InferenceSession(str(self.model_path), providers=providers)
            else:  # pytorch or quantized
                checkpoint = torch.load(self.model_path)
                model = checkpoint['model']
                model.eval()
                if self.device == 'cuda':
                    model = model.cuda()
                return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def prepare_input(self, batch_size: int, seq_length: int, num_features: int) -> torch.Tensor:
        """Prepare input tensor for inference."""
        input_tensor = torch.randn(batch_size, seq_length, num_features)
        if self.model_type == 'onnx':
            return input_tensor.numpy()
        if self.device == 'cuda':
            input_tensor = input_tensor.cuda()
        return input_tensor

    def run_inference(self, input_data: torch.Tensor) -> None:
        """Run a single inference pass."""
        if self.model_type == 'onnx':
            self.model.run(None, {'input': input_data})
        else:
            with torch.no_grad():
                self.model(input_data)

    def benchmark_configuration(self, 
                             batch_size: int, 
                             seq_length: int, 
                             num_features: int,
                             num_runs: int,
                             warmup_runs: int) -> Dict[str, float]:
        """Benchmark a specific configuration and return metrics."""
        input_data = self.prepare_input(batch_size, seq_length, num_features)
        
        # Warmup runs
        for _ in range(warmup_runs):
            self.run_inference(input_data)
        
        # Benchmark runs
        latencies = []
        memory_usage = []
        
        for _ in range(num_runs):
            # Clear cache between runs
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Measure inference time
            start = time.perf_counter()
            self.run_inference(input_data)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            
            # Measure memory usage
            if self.device == 'cuda':
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
            else:
                memory_usage.append(psutil.Process().memory_info().rss / 1024**2)  # MB
        
        return {
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'mean_memory': np.mean(memory_usage),
            'max_memory': np.max(memory_usage),
            'throughput': batch_size / np.mean(latencies)
        }

def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save benchmark results in multiple formats."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'benchmark_results.csv', index=False)
    
    # Save as JSON
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate summary report
    summary = {
        'model_type': results[0]['model_type'],
        'device': results[0]['device'],
        'avg_throughput': np.mean([r['throughput'] for r in results]),
        'max_throughput': np.max([r['throughput'] for r in results]),
        'min_latency': np.min([r['mean_latency'] for r in results]),
        'max_memory': np.max([r['max_memory'] for r in results])
    }
    
    with open(output_dir / 'benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

def main():
    args = parse_args()
    
    # Parse batch sizes and sequence lengths
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(',')]
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker(args.model_path, args.device)
    
    # Run benchmarks
    results = []
    total_configs = len(batch_sizes) * len(sequence_lengths)
    
    with tqdm(total=total_configs) as pbar:
        for batch_size in batch_sizes:
            for seq_length in sequence_lengths:
                metrics = benchmarker.benchmark_configuration(
                    batch_size=batch_size,
                    seq_length=seq_length,
                    num_features=args.num_features,
                    num_runs=args.num_runs,
                    warmup_runs=args.warmup_runs
                )
                
                results.append({
                    'model_type': benchmarker.model_type,
                    'device': args.device,
                    'batch_size': batch_size,
                    'sequence_length': seq_length,
                    'num_features': args.num_features,
                    **metrics
                })
                
                pbar.update(1)
    
    # Save results
    save_results(results, args.output_path)
    logger.info(f"Benchmark results saved to {args.output_path}")

if __name__ == '__main__':
    main() 