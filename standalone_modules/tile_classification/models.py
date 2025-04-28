"""
Modelli di classificazione per Wavelet Scattering Transform.

Questo modulo contiene le definizioni dei modelli neurali per la classificazione
di immagini utilizzando la trasformata scattering wavelet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D

class ScatteringClassifier(nn.Module):
    """Modello neurale per la classificazione con trasformata scattering."""
    
    def __init__(self, in_channels, classifier_type='cnn', num_classes=4):
        """
        Inizializza il classificatore scattering.
        
        Args:
            in_channels: Numero di canali di input (coefficienti scattering)
            classifier_type: Tipo di classificatore ('cnn', 'mlp', o 'linear')
            num_classes: Numero di classi di output
        """
        super(ScatteringClassifier, self).__init__()
        self.in_channels = in_channels
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.build()

    def build(self):
        """Costruisce l'architettura del classificatore in base al tipo."""
        self.K = self.in_channels
        self.bn = nn.BatchNorm2d(self.K)
        
        if self.classifier_type == 'cnn':
            # Classificatore CNN con strati convoluzionali profondi
            cfg = [256, 256, 256, 'M', 512, 512, 512, 1024, 1024]
            layers = []
            current_in_channels = self.K
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [
                        nn.Conv2d(current_in_channels, v, kernel_size=3, padding=1),
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                    current_in_channels = v
            layers += [nn.AdaptiveAvgPool2d(2)]
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(1024 * 4, self.num_classes)
        elif self.classifier_type == 'mlp':
            # Classificatore MLP
            self.classifier = nn.Sequential(
                nn.Linear(self.K * 8 * 8, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.num_classes))
            self.features = None
        elif self.classifier_type == 'linear':
            # Classificatore lineare
            self.classifier = nn.Linear(self.K * 8 * 8, self.num_classes)
            self.features = None

    def forward(self, x):
        """
        Forward pass attraverso la rete.
        
        Args:
            x: Tensore di input di forma (batch_size, in_channels, 8, 8)
            
        Returns:
            Tensore di output di forma (batch_size, num_classes)
        """
        # Ottieni la dimensione del batch e ridimensiona direttamente usando i canali di input attesi (self.K)
        batch_size = x.size(0)
        total_elements = x.numel()
        
        # Se l'input ha già la forma corretta, usalo direttamente
        if x.shape[1] == self.K and x.shape[2] == 8 and x.shape[3] == 8:
            pass
        # Se l'input proviene da una trasformata scattering e deve essere ridimensionato
        elif total_elements // batch_size // 64 == self.K:
            # Già il numero corretto di elementi
            x = x.view(batch_size, self.K, 8, 8)
        else:
            # Gestisci il caso in cui l'output della scattering ha un numero diverso di canali
            # Questo è specificamente per il caso in cui l'output della scattering ha forma [1, 3, 81, 8, 8]
            # e deve essere appiattito per corrispondere ai canali
            flattened = x.reshape(batch_size, -1)
            if flattened.shape[1] % 64 == 0:  # Può essere ridimensionato a [batch, channels, 8, 8]
                # Prendi i primi self.K * 64 elementi e ridimensiona
                x = flattened[:, :self.K * 64].view(batch_size, self.K, 8, 8)
            else:
                raise ValueError(f"La forma dell'input {x.shape} con {flattened.shape[1]} elementi non può essere ridimensionata a [batch, {self.K}, 8, 8]")
                
        x = self.bn(x)
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def create_scattering_transform(J=2, shape=(32, 32), max_order=2, device=None):
    """
    Crea una trasformata scattering.
    
    Args:
        J: Numero di scale
        shape: Forma delle immagini di input
        max_order: Ordine massimo della scattering
        device: Device su cui creare la trasformata scattering
        
    Returns:
        Oggetto Scattering2D
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    scattering = Scattering2D(
        J=J,
        shape=shape,
        max_order=max_order
    ).to(device)
    
    return scattering

def create_classification_model(config):
    """
    Crea un modello di classificazione scattering.
    
    Args:
        config: Oggetto di configurazione con i parametri del modello
        
    Returns:
        Modello ScatteringClassifier e trasformata Scattering2D
    """
    # Crea la trasformata scattering
    scattering = create_scattering_transform(
        J=config.J,
        shape=config.shape,
        max_order=config.scattering_order,
        device=config.device
    )
    
    # Crea il modello di classificazione
    model = ScatteringClassifier(
        in_channels=config.scattering_coeffs,
        classifier_type='cnn',
        num_classes=config.num_classes
    ).to(config.device)
    
    return model, scattering

def print_classifier_summary(model, scattering, device, input_shape=(1, 3, 32, 32)):
    """
    Stampa un riepilogo del modello di classificazione.
    
    Args:
        model: Modello ScatteringClassifier
        scattering: Trasformata Scattering2D
        device: Device da utilizzare
        input_shape: Forma delle immagini di input
    """
    print("\n" + "="*80)
    print(" "*30 + "RIEPILOGO DEL MODELLO DI CLASSIFICAZIONE" + " "*30)
    print("="*80)
    
    # Crea un input fittizio
    dummy_input = torch.zeros(input_shape).to(device)
    
    # Ottieni la forma dell'output della scattering
    with torch.no_grad():
        scat_output = scattering(dummy_input)
        scat_shape = scat_output.shape
    
    print("\nTRASFORMATA SCATTERING:")
    print(f"  • Forma dell'input: {input_shape}")
    print(f"  • Parametro J: {scattering.J}")
    print(f"  • Ordine massimo: {scattering.max_order}")
    print(f"  • Forma dell'output: {scat_shape}")
    
    print("\nARCHITETTURA DEL MODELLO:")
    print(f"  • Tipo: {model.classifier_type}")
    print(f"  • Canali di input: {model.in_channels}")
    print(f"  • Numero di classi: {model.num_classes}")
    print(f"  • Parametri: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nLIVELLI:")
    if model.features:
        print("  • Batch Normalization")
        print("  • Estrattore di caratteristiche (livelli CNN)")
        for idx, layer in enumerate(model.features):
            print(f"    - {idx}: {layer}")
    else:
        print("  • Batch Normalization")
    
    print("  • Classificatore:")
    if isinstance(model.classifier, nn.Linear):
        print(f"    - Linear: {model.classifier}")
    else:
        for idx, layer in enumerate(model.classifier):
            print(f"    - {idx}: {layer}")
    
    print("\n" + "="*80)
