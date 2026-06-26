import torch 
import torchvision.models as models # models library
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

def load_model():
    print("Loading ResNet-18 pretrained model...") 
    model = models.resnet18(weights='IMAGENET1K_V1') # Weights decides the weight t
                                                     #to download
    model.eval() # pythorch defined function, puts the model into evaluation mode
    return model


def export_to_onnx(model, onnx_path):
    print(f"Exporting model to {onnx_path}...") 
    dummy_input = torch.randn(1, 3, 224, 224) # creates an example of the input
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=18,
        dynamo=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_path}")

    
def verify_export(model, onnx_path):
    print("Verifying ONNX export...")

    # Verificar que el archivo ONNX es válido
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Crear input de prueba
    test_input = torch.randn(1, 3, 224, 224)

    # Output de PyTorch
    with torch.no_grad():
        pytorch_output = model(test_input).numpy()

    # Output de ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    ort_output = ort_session.run(
        None,
        {'input': test_input.numpy()}
    )[0]

    # Comparar outputs
    np.testing.assert_allclose(pytorch_output, ort_output, rtol=1e-3, atol=1e-5)
    print("Verification passed! PyTorch and ONNX Runtime outputs match.")

def main():
    # Rutas
    onnx_path = Path("models/resnet18.onnx")
    
    # Cargar modelo
    model = load_model()
    
    # Exportar a ONNX
    export_to_onnx(model, onnx_path)
    
    # Verificar exportación
    verify_export(model, onnx_path)
    
    # Info del modelo exportado
    model_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")

if __name__ == "__main__":
    main()