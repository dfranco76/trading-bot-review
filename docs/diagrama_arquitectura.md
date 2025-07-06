# Diagrama de Arquitectura

```mermaid
graph TB
    A[Market Data API] --> B[Data Module]
    B --> C{Sistema Multi-Agente}
    
    C --> D[Agente Momentum]
    C --> E[Agente Mean Reversion]
    C --> F[Agente Pattern Recognition]
    C --> G[Agente ML Prediction]
    
    D --> H[Señales Consolidadas]
    E --> H
    F --> H
    G --> H
    
    H --> I[Risk Manager]
    I --> J{Validación}
    
    J -->|Aprobado| K[Execution Module]
    J -->|Rechazado| L[Log Rechazo]
    
    K --> M[Broker API]
    M --> N[Mercado]
    
    K --> O[Monitoring System]
    I --> O
    B --> O
    
    O --> P[Dashboard]
    O --> Q[Alertas]
    O --> R[Reports]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style N fill:#9f9,stroke:#333,stroke-width:2px
    style I fill:#ff9,stroke:#333,stroke-width:2px