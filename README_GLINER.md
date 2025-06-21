# GLiNER Integration for Tourism Crawler

## Overview

GLiNER (Generalist Model for Named Entity Recognition) has been integrated into the tourism crawler to extract entities from web content before inserting them into the ChromaDB database. GLiNER is a zero-shot NER model that can identify entities without being limited to predefined entity types.

## Features

### Entity Types Extracted

The GLiNER agent is configured to extract the following tourism-related entities in both Spanish and English:

#### Location Entities
- **Countries** (PAÍS/COUNTRY)
- **Cities** (CIUDAD/CITY)
- **Regions** (REGIÓN/REGION)
- **Tourist Destinations** (DESTINO_TURÍSTICO/TOURIST_DESTINATION)
- **Beaches** (PLAYA/BEACH)
- **Mountains** (MONTAÑA/MOUNTAIN)
- **Islands** (ISLA/ISLAND)
- **Towns** (PUEBLO/TOWN)

#### Accommodation
- **Hotels** (HOTEL)
- **Resorts** (RESORT)
- **Hostels** (HOSTAL/HOSTEL)
- **Accommodations** (ALOJAMIENTO/ACCOMMODATION)
- **Hotel Chains** (CADENA_HOTELERA/HOTEL_CHAIN)

#### Attractions & Points of Interest
- **Tourist Attractions** (ATRACCIÓN_TURÍSTICA/TOURIST_ATTRACTION)
- **Museums** (MUSEO/MUSEUM)
- **Monuments** (MONUMENTO/MONUMENT)
- **Parks** (PARQUE/PARK)
- **Historical Sites** (SITIO_HISTÓRICO/HISTORICAL_SITE)
- **Heritage Sites** (PATRIMONIO/HERITAGE_SITE)
- **Shopping Centers** (CENTRO_COMERCIAL/SHOPPING_CENTER)
- **Markets** (MERCADO/MARKET)

#### Services & Activities
- **Restaurants** (RESTAURANTE/RESTAURANT)
- **Bars** (BAR)
- **Cafes** (CAFÉ/CAFE)
- **Tours** (TOUR)
- **Excursions** (EXCURSIÓN/EXCURSION)
- **Activities** (ACTIVIDAD/ACTIVITY)
- **Events** (EVENTO/EVENT)
- **Festivals** (FESTIVAL)

#### Transportation
- **Airports** (AEROPUERTO/AIRPORT)
- **Stations** (ESTACIÓN/STATION)
- **Ports** (PUERTO/PORT)
- **Airlines** (LÍNEA_AÉREA/AIRLINE)
- **Cruises** (CRUCERO/CRUISE)

#### Practical Information
- **Prices** (PRECIO/PRICE)
- **Currencies** (MONEDA/CURRENCY)
- **Schedules** (HORARIO/SCHEDULE)
- **Seasons** (TEMPORADA/SEASON)
- **Weather** (CLIMA/WEATHER)

#### Organizations
- **Travel Agencies** (AGENCIA_VIAJES/TRAVEL_AGENCY)
- **Tour Operators** (OPERADOR_TURÍSTICO/TOUR_OPERATOR)
- **Tourism Offices** (OFICINA_TURISMO/TOURISM_OFFICE)

## Installation

1. Install the required dependencies:
```bash
pip install gliner torch transformers
```

2. The GLiNER model will be automatically downloaded when first used. The default model is `urchade/gliner_multi-v2.1` which supports multiple languages.

## Usage

### Basic Usage with GLiNER

```python
from crawler import TourismCrawler

# Create crawler instance with Gemini disabled
crawler = TourismCrawler(
    starting_urls=["https://www.example-tourism-site.com"],
    chroma_collection_name="tourism_data_gliner",
    max_pages=50,
    enable_gemini_processing=False  # Disable Gemini
)

# Enable GLiNER processing
crawler.enable_gliner()

# Run the crawler
pages_added = crawler.run_parallel_crawler()
```

### Using GLiNER with Keyword Search

```python
# Create crawler for keyword-based search
crawler = TourismCrawler(
    starting_urls=[],
    chroma_collection_name="tourism_gliner_search",
    max_pages=30,
    enable_gemini_processing=False
)

# Enable GLiNER
crawler.enable_gliner()

# Search and crawl with keywords
keywords = ["beach hotels caribbean", "all inclusive resorts"]
pages_added = crawler.run_parallel_crawler_from_keywords(
    keywords=keywords,
    max_depth=2
)
```

### Switching Between Processors

You can switch between Gemini and GLiNER processors:

```python
# Start with Gemini
crawler = TourismCrawler(
    starting_urls=urls,
    enable_gemini_processing=True
)

# Later, switch to GLiNER
crawler.disable_gemini()
crawler.enable_gliner()
```

## Data Structure

### Processed Data Format

When GLiNER processes content, it structures the data as follows:

```json
{
    "source_url": "https://example.com/page",
    "source_title": "Page Title",
    "processed_by": "gliner",
    "summary": "Generated summary of entities found",
    "entities": {
        "countries": ["Cuba", "Mexico"],
        "cities": ["Havana", "Varadero"],
        "hotels": [
            {"name": "Hotel Nacional", "type": "hotel"},
            {"name": "Meliá Varadero", "type": "resort"}
        ],
        "attractions": [
            {"name": "Old Havana", "type": "historical_site"},
            {"name": "Varadero Beach", "type": "beach"}
        ],
        "restaurants": [
            {"name": "La Bodeguita del Medio", "type": "restaurant"}
        ],
        "prices": [
            {"text": "$150 per night", "type": "price"},
            {"text": "25 CUC entrance fee", "type": "price"}
        ]
    },
    "raw_entities": [
        {"text": "Cuba", "type": "COUNTRY", "confidence": 0.95},
        {"text": "Hotel Nacional", "type": "HOTEL", "confidence": 0.89}
    ]
}
```

### ChromaDB Metadata

The processed data is stored in ChromaDB with the following metadata:

- `url`: Source URL
- `title`: Page title
- `source`: "parallel_tourism_crawler"
- `processed_by_gliner`: true
- `entities_data`: Full JSON of extracted entities
- `countries`: Comma-separated list of countries
- `cities`: Comma-separated list of cities
- `hotels`: Comma-separated list of hotel names (max 5)

## Performance Considerations

1. **Model Loading**: GLiNER model is loaded once when enabled, which may take a few seconds.

2. **Processing Speed**: GLiNER is generally faster than Gemini API calls but slower than raw text extraction.

3. **Memory Usage**: The model requires approximately 1-2GB of RAM.

4. **Accuracy**: GLiNER provides good zero-shot entity recognition but may not capture complex relationships like Gemini.

## Comparison with Gemini

| Feature | GLiNER | Gemini |
|---------|---------|---------|
| Speed | Fast (local) | Slower (API calls) |
| Cost | Free | API costs |
| Entity Types | Predefined labels | Flexible extraction |
| Structured Output | Entity-focused | Full JSON structure |
| Relationship Extraction | Limited | Advanced |
| Language Support | Multilingual | Multilingual |
| Offline Capability | Yes | No |

## Testing

Run the test script to see GLiNER in action:

```bash
python test_gliner.py
```

This provides three test options:
1. Basic test with predefined URLs
2. Keyword-based search test
3. Comparison between Gemini and GLiNER processors

## Troubleshooting

### Common Issues

1. **Model Download Failed**
   - Check internet connection
   - Manually download the model from Hugging Face

2. **Out of Memory**
   - Reduce batch size
   - Use a smaller model variant
   - Increase system RAM

3. **Low Entity Detection**
   - Adjust confidence threshold (default: 0.5)
   - Try different entity labels
   - Ensure text is clean and well-formatted

### Debug Output

The crawler provides detailed output for GLiNER processing:

```
📝 GUARDANDO CHUNK EN CHROMADB:
   📌 ID: gliner_doc_1234567_0_1234567890
   🔗 URL: https://example.com
   📄 Título: Tourism Page Title...
   📏 Tamaño del texto: 1523 caracteres
   🏷️ Procesado por: GLiNER
   🌍 Países: Cuba, Mexico
   🏙️ Ciudades: Havana, Cancun
   📊 Metadata: 8 campos
   ✅ Chunk guardado exitosamente
```

## Future Improvements

1. **Custom Entity Types**: Add domain-specific entity types for tourism
2. **Confidence Tuning**: Implement adaptive confidence thresholds
3. **Relationship Extraction**: Enhance to capture entity relationships
4. **Caching**: Implement entity caching for repeated content
5. **Model Fine-tuning**: Fine-tune GLiNER on tourism-specific data