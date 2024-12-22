import argparse
import torch
import torch.quantization
import logging
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Quantize SSM_HIPPO model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the quantized model')
    parser.add_argument('--quantization_type', type=str, default='dynamic',
                       choices=['dynamic', 'static'],
                       help='Type of quantization to apply')
    parser.add_argument('--calibration_data', type=str,
                       help='Path to calibration data (required for static quantization)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for calibration')
    return parser.parse_args()

def load_model(model_path: str) -> Tuple[torch.nn.Module, dict]:
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

def prepare_for_quantization(model: torch.nn.Module) -> torch.nn.Module:
    """Prepare model for quantization by adding quantization observers."""
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    return torch.quantization.prepare(model)

def calibrate_model(model: torch.nn.Module, 
                   calibration_data: str,
                   batch_size: int) -> None:
    """Calibrate the model using provided calibration data."""
    try:
        # Load calibration data
        calib_data = torch.load(calibration_data)
        
        # Run calibration
        with torch.no_grad():
            for i in range(0, len(calib_data), batch_size):
                batch = calib_data[i:i + batch_size]
                _ = model(batch)
                
        logger.info("Calibration completed successfully")
    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        raise

def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic quantization to the model."""
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},  # Quantize linear and LSTM layers
            dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        logger.error(f"Dynamic quantization failed: {str(e)}")
        raise

def quantize_static(model: torch.nn.Module) -> torch.nn.Module:
    """Apply static quantization to the model."""
    try:
        return torch.quantization.convert(model)
    except Exception as e:
        logger.error(f"Static quantization failed: {str(e)}")
        raise

def save_quantized_model(model: torch.nn.Module,
                        config: dict,
                        output_path: str) -> None:
    """Save the quantized model and its configuration."""
    try:
        torch.save({
            'model': model,
            'config': config,
            'quantized': True
        }, output_path)
        logger.info(f"Quantized model saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save quantized model: {str(e)}")
        raise

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    model, config = load_model(args.model_path)
    
    if args.quantization_type == 'dynamic':
        logger.info("Applying dynamic quantization...")
        quantized_model = quantize_dynamic(model)
    else:  # static quantization
        if not args.calibration_data:
            raise ValueError("Calibration data is required for static quantization")
        
        logger.info("Preparing model for static quantization...")
        model = prepare_for_quantization(model)
        
        logger.info("Calibrating model...")
        calibrate_model(model, args.calibration_data, args.batch_size)
        
        logger.info("Applying static quantization...")
        quantized_model = quantize_static(model)
    
    # Save quantized model
    save_quantized_model(quantized_model, config, args.output_path)
    
    # Log model size reduction
    original_size = Path(args.model_path).stat().st_size / (1024 * 1024)  # MB
    quantized_size = Path(args.output_path).stat().st_size / (1024 * 1024)  # MB
    reduction = (original_size - quantized_size) / original_size * 100
    
    logger.info(f"Original model size: {original_size:.2f} MB")
    logger.info(f"Quantized model size: {quantized_size:.2f} MB")
    logger.info(f"Size reduction: {reduction:.2f}%")

if __name__ == '__main__':
    main() 