# Sistema Multiagente para Turismo Inteligente

Sistema inteligente de asistencia turística basado en agentes que utiliza técnicas de IA para proporcionar recomendaciones personalizadas, planificación de viajes y simulación de experiencias turísticas.

**Grupo:** C312

**Integrantes:**
- Gabriel Herrera Carrazana
- Adrian A Souto Morales  
- Lauren Peraza García

## Descripción

Sistema inteligente de asistencia turística basado en agentes que utiliza técnicas de IA para proporcionar recomendaciones personalizadas, planificación de viajes y simulación de experiencias turísticas.

## Estructura del Proyecto

```
Proyecto-IA-SIM-SRI/
│
├── agents/                 
│   ├── agent_coordinator.py    
│   ├── agent_crawler.py        
│   ├── agent_rag.py           
│   ├── agent_interface.py      
│   ├── agent_context.py        
│   ├── agent_route.py          
│   ├── agent_tourist_guide.py  
│   ├── agent_simulation.py     
│   ├── agent_processor.py      
│   └── agent_gliner.py         
│
├── core/                   
│   ├── crawler.py             
│   ├── rag.py                 
│   ├── chromadb_singleton.py  
│   └── mistral_config.py      
│
├── utils/                  
│   ├── urls.py               
│   ├── simulation_utils.py   
│   ├── ant_colony_crawler.py 
│   └── fix_flaml_warning.py  
│
├── tests/                  
│   ├── test_preferences_*.py  
│   └── test_gliner.py        
│
├── docs/                   
│   ├── CAMBIOS_*.md          
│   ├── FIX_*.md              
│   └── README_GLINER.md      
│
├── logs/                   
│   └── crawler_logs/         
│
├── main.py                 
├── requirements.txt        
├── .env                   
└── README.md              
```




- **Agente Coordinador**: Orquesta la comunicación entre todos los agentes
- **Agente Crawler**: Recopila información turística de la web con crawling paralelo
- **Agente RAG**: Genera respuestas basadas en la información recopilada
- **Agente de Contexto**: Mantiene el contexto conversacional
- **Agente de Rutas**: Optimiza rutas turísticas
- **Agente Guía Turístico**: Asistente especializado en planificación de viajes
- **Agente de Simulación**: Simula experiencias turísticas con lógica difusa


- **Mistral AI**: Para generación de respuestas y procesamiento de lenguaje natural
- **ChromaDB**: Base de datos vectorial para almacenamiento eficiente
- **GLiNER**: Extracción de entidades nombradas
- **Algoritmos Genéticos**: Optimización de selección de documentos
- **ACO (Ant Colony Optimization)**: Crawling inteligente de páginas web


- Planificación personalizada de vacaciones
- Búsqueda inteligente de información turística
- Optimización de rutas turísticas
- Simulación de experiencias de viaje
- Contexto conversacional persistente
- Crawling paralelo con múltiples hilos



- Python 3.8+
- API Key de Mistral AI
- Dependencias listadas en `requirements.txt`



1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/Proyecto-IA-SIM-SRI.git
cd Proyecto-IA-SIM-SRI
```

2. Crear entorno virtual:
```bash
python -m venv .venv
.venv\Scripts\activate  
source .venv/bin/activate  
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
```bash

echo "GOOGLE_API_KEY=tu_api_key_aqui" > .env
```




```bash
python main.py
```


- `stats` - Ver estadísticas de conversación
- `contexto` - Ver historial de conversación
- `limpiar` - Limpiar contexto de conversación
- `salir` - Terminar el programa
- `cancelar` - Salir del modo planificación



Ejecutar todas las pruebas:
```bash
python -m pytest tests/
```

Ejecutar prueba específica:
```bash
python tests/test_preferences_simple.py
```




1. Usuario hace una consulta
2. Agente Coordinador analiza la intención
3. Si es necesario, el Crawler busca información nueva
4. El sistema RAG genera una respuesta
5. El Agente de Contexto guarda la interacción
6. Se devuelve la respuesta al usuario


- **ChromaDB**: Almacena embeddings de documentos
- **Persistencia**: Los datos se guardan en `chroma_db/`
- **Logs**: Se guardan en `logs/crawler_logs/`




- Búsqueda en múltiples motores (DuckDuckGo, Bing, Searx)
- Filtrado por relevancia turística
- Procesamiento paralelo con múltiples hilos
- Extracción estructurada de información


- Algoritmos genéticos para selección de documentos
- Similitud coseno para relevancia
- Contexto conversacional para mejores respuestas


- Lógica difusa para simular satisfacción
- Consideración de múltiples factores (clima, tiempo, preferencias)
- Visualización de resultados



Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request



Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.



- Sistema desarrollado como proyecto de IA para simulación y sistemas de recuperación de información



- Mistral AI por proporcionar capacidades de generación de lenguaje
- ChromaDB por la base de datos vectorial
- La comunidad de Python por las excelentes librerías

