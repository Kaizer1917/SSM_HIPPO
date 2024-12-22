import argparse
import torch
import onnx
import onnxruntime
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export SSM_HIPPO model to ONNX')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the ONNX model')
    parser.add_argument('--input_shape', type=str, default='1,32,256',
                       help='Input shape in format batch,sequence_length,features')
    parser.add_argument('--optimize', action='store_true',
                       help='Apply ONNX optimizations')
    parser.add_argument('--dynamic_axes', action='store_true',
                       help='Enable dynamic axes for variable length inputs')
    parser.add_argument('--opset_version', type=int, default=13,
                       help='ONNX opset version')
    return parser.parse_args()

def load_model(model_path: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load the model and its configuration from checkpoint."""
    try:
        checkpoint = torch.load(model_path)
        model = checkpoint['model']
        config = checkpoint.get('config', {})
        model.eval()
        return model, config
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

def prepare_dummy_input(input_shape: str) -> torch.Tensor:
    """Create dummy input tensor based on provided shape."""
    try:
        shape = [int(x) for x in input_shape.split(',')]
        return torch.randn(*shape)
    except Exception as e:
        logger.error(f"Failed to create dummy input: {str(e)}")
        raise

def get_dynamic_axes(seq_len: int) -> Dict[str, Dict[int, str]]:
    """Define dynamic axes for variable length inputs and outputs."""
    return {
        'input': {
            0: 'batch_size',
            1: 'sequence_length'
        },
        'output': {
            0: 'batch_size',
            1: 'sequence_length'
        }
    }

def optimize_onnx_model(model_path: str) -> None:
    """Apply ONNX optimizations to the exported model."""
    try:
        model = onnx.load(model_path)
        
        # Basic model checks
        onnx.checker.check_model(model)
        
        # Optimize
        from onnxruntime.transformers import optimizer
        opt_model = optimizer.optimize_model(
            model_path,
            model_type='bert',  # Using bert type as it has good general optimizations
            num_heads=8,  # Adjust based on your model
            hidden_size=256  # Adjust based on your model
        )
        
        # Save optimized model
        opt_model.save_model_to_file(model_path)
        logger.info("Model optimization completed successfully")
    except Exception as e:
        logger.error(f"Model optimization failed: {str(e)}")
        raise

def verify_onnx_model(onnx_path: str, 
                     pytorch_model: torch.nn.Module,
                     dummy_input: torch.Tensor) -> None:
    """Verify ONNX model outputs match PyTorch model outputs."""
    try:
        # Get PyTorch model prediction
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input)

        # Get ONNX model prediction
        ort_session = onnxruntime.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # Compare outputs
        np.testing.assert_allclose(pytorch_output.numpy(), 
                                 ort_output, 
                                 rtol=1e-03, 
                                 atol=1e-05)
        logger.info("ONNX model verification successful")
    except Exception as e:
        logger.error(f"ONNX model verification failed: {str(e)}")
        raise

def export_to_onnx(model: torch.nn.Module,
                  dummy_input: torch.Tensor,
                  output_path: str,
                  dynamic_axes: Dict[str, Dict[int, str]] = None,
                  opset_version: int = 13) -> None:
    """Export PyTorch model to ONNX format."""
    try:
        torch.onnx.export(model,
                         dummy_input,
                         output_path,
                         export_params=True,
                         opset_version=opset_version,
                         do_constant_folding=True,
                         input_names=['input'],
                         output_names=['output'],
                         dynamic_axes=dynamic_axes)
        logger.info(f"Model exported to ONNX format at {output_path}")
    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}")
        raise

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading PyTorch model...")
    model, config = load_model(args.model_path)
    
    # Prepare dummy input
    dummy_input = prepare_dummy_input(args.input_shape)
    
    # Prepare dynamic axes if enabled
    dynamic_axes = get_dynamic_axes(int(args.input_shape.split(',')[1])) if args.dynamic_axes else None
    
    # Export to ONNX
    logger.info("Exporting to ONNX...")
    export_to_onnx(
        model,
        dummy_input,
        args.output_path,
        dynamic_axes,
        args.opset_version
    )
    
    # Verify the exported model
    logger.info("Verifying ONNX model...")
    verify_onnx_model(args.output_path, model, dummy_input)
    
    # Optimize if requested
    if args.optimize:
        logger.info("Optimizing ONNX model...")
        optimize_onnx_model(args.output_path)
    
    # Log model details
    original_size = Path(args.model_path).stat().st_size / (1024 * 1024)  # MB
    onnx_size = Path(args.output_path).stat().st_size / (1024 * 1024)  # MB
    
    logger.info(f"Original model size: {original_size:.2f} MB")
    logger.info(f"ONNX model size: {onnx_size:.2f} MB")
    logger.info("ONNX export completed successfully")

if __name__ == '__main__':
    main() 