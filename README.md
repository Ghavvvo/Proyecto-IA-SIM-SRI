# Sistema Multiagente de Turismo con IA

Sistema inteligente de asistencia turística basado en agentes que utiliza técnicas de IA para proporcionar recomendaciones personalizadas, planificación de viajes y simulación de experiencias turísticas.

## 🏗️ Estructura del Proyecto

```
Proyecto-IA-SIM-SRI/
│
├── agents/                 # Agentes del sistema
│   ├── agent_coordinator.py    # Coordinador principal
│   ├── agent_crawler.py        # Agente de crawling web
│   ├── agent_rag.py           # Agente RAG (Retrieval-Augmented Generation)
│   ├── agent_interface.py      # Interfaz de usuario
│   ├── agent_context.py        # Gestión de contexto conversacional
│   ├── agent_route.py          # Optimización de rutas
│   ├── agent_tourist_guide.py  # Guía turístico virtual
│   ├── agent_simulation.py     # Simulación de experiencias
│   ├── agent_processor.py      # Procesamiento con Gemini
│   └── agent_gliner.py         # Extracción de entidades con GLiNER
│
├── core/                   # Componentes principales
│   ├── crawler.py             # Motor de crawling paralelo
│   ├── rag.py                 # Sistema RAG mejorado
│   ├── chromadb_singleton.py  # Gestión de base de datos vectorial
│   └── gemini_config.py       # Configuración de Gemini AI
│
├── utils/                  # Utilidades
│   ├── urls.py               # URLs de inicio para crawling
│   ├── simulation_utils.py   # Utilidades de simulación
│   ├── ant_colony_crawler.py # Algoritmo ACO para crawling
│   └── fix_flaml_warning.py  # Corrección de warnings
│
├── tests/                  # Pruebas unitarias
│   ├── test_preferences_*.py  # Tests de preferencias
│   └── test_gliner.py        # Test de GLiNER
│
├── docs/                   # Documentación
│   ├── CAMBIOS_*.md          # Logs de cambios
│   ├── FIX_*.md              # Documentación de correcciones
│   └── README_GLINER.md      # Documentación de GLiNER
│
├── logs/                   # Archivos de log
│   └── crawler_logs/         # Logs del crawler
│
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
├── .env                   # Variables de entorno
└── README.md              # Este archivo
```

## 🚀 Características Principales

### 1. **Sistema Multiagente**
- **Agente Coordinador**: Orquesta la comunicación entre todos los agentes
- **Agente Crawler**: Recopila información turística de la web con crawling paralelo
- **Agente RAG**: Genera respuestas basadas en la información recopilada
- **Agente de Contexto**: Mantiene el contexto conversacional
- **Agente de Rutas**: Optimiza rutas turísticas
- **Agente Guía Turístico**: Asistente especializado en planificación de viajes
- **Agente de Simulación**: Simula experiencias turísticas con lógica difusa

### 2. **Tecnologías de IA**
- **Gemini AI**: Para generación de respuestas y procesamiento de lenguaje natural
- **ChromaDB**: Base de datos vectorial para almacenamiento eficiente
- **GLiNER**: Extracción de entidades nombradas
- **Algoritmos Genéticos**: Optimización de selección de documentos
- **ACO (Ant Colony Optimization)**: Crawling inteligente de páginas web

### 3. **Funcionalidades**
- Planificación personalizada de vacaciones
- Búsqueda inteligente de información turística
- Optimización de rutas turísticas
- Simulación de experiencias de viaje
- Contexto conversacional persistente
- Crawling paralelo con múltiples hilos

## 📋 Requisitos

- Python 3.8+
- API Key de Google Gemini
- Dependencias listadas en `requirements.txt`

## 🔧 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/Proyecto-IA-SIM-SRI.git
cd Proyecto-IA-SIM-SRI
```

2. Crear entorno virtual:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
```bash
# Crear archivo .env
echo "GOOGLE_API_KEY=tu_api_key_aqui" > .env
```

## 🎮 Uso

### Ejecución básica:
```bash
python main.py
```

### Comandos disponibles durante la conversación:
- `stats` - Ver estadísticas de conversación
- `contexto` - Ver historial de conversación
- `limpiar` - Limpiar contexto de conversación
- `salir` - Terminar el programa
- `cancelar` - Salir del modo planificación

## 🧪 Pruebas

Ejecutar todas las pruebas:
```bash
python -m pytest tests/
```

Ejecutar prueba específica:
```bash
python tests/test_preferences_simple.py
```

## 📊 Arquitectura del Sistema

### Flujo de Datos
1. Usuario hace una consulta
2. Agente Coordinador analiza la intención
3. Si es necesario, el Crawler busca información nueva
4. El sistema RAG genera una respuesta
5. El Agente de Contexto guarda la interacción
6. Se devuelve la respuesta al usuario

### Base de Datos
- **ChromaDB**: Almacena embeddings de documentos
- **Persistencia**: Los datos se guardan en `chroma_db/`
- **Logs**: Se guardan en `logs/crawler_logs/`

## 🔍 Características Avanzadas

### Crawling Inteligente
- Búsqueda en múltiples motores (DuckDuckGo, Bing, Searx)
- Filtrado por relevancia turística
- Procesamiento paralelo con múltiples hilos
- Extracción estructurada de información

### Optimización de Respuestas
- Algoritmos genéticos para selección de documentos
- Similitud coseno para relevancia
- Contexto conversacional para mejores respuestas

### Simulación de Experiencias
- Lógica difusa para simular satisfacción
- Consideración de múltiples factores (clima, tiempo, preferencias)
- Visualización de resultados

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## 👥 Autores

- Sistema desarrollado como proyecto de IA para simulación y sistemas de recuperación de información

## 🙏 Agradecimientos

- Google Gemini AI por proporcionar capacidades de generación de lenguaje
- ChromaDB por la base de datos vectorial
- La comunidad de Python por las excelentes librerías